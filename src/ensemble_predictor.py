"""
Ensemble Match Predictor - Combine XGBoost + Random Forest for better predictions
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from typing import Dict, List
import joblib


class EnsemblePredictor:
    """
    Ensemble model combining multiple algorithms for match prediction
    """
    
    def __init__(self):
        self.ensemble = None
        self.xgb_model = None
        self.rf_model = None
        self.lr_model = None
        self.feature_columns = None
        self.label_mapping = {
            'home_win': 0,
            'draw': 1,
            'away_win': 2
        }
        self.inverse_label_mapping = {v: k for k, v in self.label_mapping.items()}
    
    def create_models(self, xgb_params: Dict = None, rf_params: Dict = None):
        """
        Create individual models with specified parameters
        """
        # XGBoost parameters
        if xgb_params is None:
            xgb_params = {
                'objective': 'multi:softprob',
                'num_class': 3,
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 150,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 3,
                'gamma': 0.1,
                'random_state': 42
            }
        
        # Random Forest parameters
        if rf_params is None:
            rf_params = {
                'n_estimators': 200,
                'max_depth': 12,
                'min_samples_split': 10,
                'min_samples_leaf': 4,
                'max_features': 'sqrt',
                'random_state': 42,
                'n_jobs': -1
            }
        
        # Create models
        self.xgb_model = xgb.XGBClassifier(**xgb_params)
        self.rf_model = RandomForestClassifier(**rf_params)
        
        # Logistic Regression for baseline (uses fewer features)
        self.lr_model = LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    
    def select_features_for_models(self, X: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Select optimal features for each model type
        """
        all_features = X.copy()
        
        # XGBoost: Use all features (handles interactions well)
        xgb_features = all_features
        
        # Random Forest: Use top features (prevent overfitting)
        # For initial training, use all; can optimize later
        rf_features = all_features
        
        # Logistic Regression: Linear features only
        linear_feature_names = [col for col in all_features.columns 
                               if any(x in col.lower() for x in ['elo', 'rating', 'goals', 'possession'])]
        lr_features = all_features[linear_feature_names] if linear_feature_names else all_features
        
        return {
            'xgb': xgb_features,
            'rf': rf_features,
            'lr': lr_features
        }
    
    def train(self, features_df: pd.DataFrame, test_size: float = 0.2,
              ensemble_weights: List[float] = None) -> Dict:
        """
        Train ensemble model with all base models
        
        Args:
            features_df: DataFrame with features and outcomes
            test_size: Proportion for testing
            ensemble_weights: Weights for voting [xgb, rf, lr]. Default: [0.5, 0.3, 0.2]
        
        Returns:
            Dictionary with training metrics
        """
        if ensemble_weights is None:
            ensemble_weights = [0.5, 0.3, 0.2]  # Favor XGBoost
        
        # Prepare data
        X = features_df.drop(['outcome', 'match_id', 'home_team_id', 'away_team_id'], 
                            axis=1, errors='ignore')
        y = features_df['outcome']
        
        # Fill NaN values (XGBoost handles them, but RF and LR don't)
        X = X.fillna(0)
        
        self.feature_columns = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Create models
        self.create_models()
        
        # Get model-specific features
        feature_sets = self.select_features_for_models(X_train)
        test_feature_sets = self.select_features_for_models(X_test)
        
        print("Training individual models...")
        
        # Train XGBoost
        print("  - Training XGBoost...")
        self.xgb_model.fit(feature_sets['xgb'], y_train)
        xgb_pred = self.xgb_model.predict(test_feature_sets['xgb'])
        xgb_acc = accuracy_score(y_test, xgb_pred)
        print(f"    XGBoost accuracy: {xgb_acc:.3f}")
        
        # Train Random Forest
        print("  - Training Random Forest...")
        self.rf_model.fit(feature_sets['rf'], y_train)
        rf_pred = self.rf_model.predict(test_feature_sets['rf'])
        rf_acc = accuracy_score(y_test, rf_pred)
        print(f"    Random Forest accuracy: {rf_acc:.3f}")
        
        # Train Logistic Regression
        print("  - Training Logistic Regression...")
        self.lr_model.fit(feature_sets['lr'], y_train)
        lr_pred = self.lr_model.predict(test_feature_sets['lr'])
        lr_acc = accuracy_score(y_test, lr_pred)
        print(f"    Logistic Regression accuracy: {lr_acc:.3f}")
        
        # Create voting ensemble
        print("\n  - Creating ensemble...")
        self.ensemble = VotingClassifier(
            estimators=[
                ('xgb', self.xgb_model),
                ('rf', self.rf_model),
                ('lr', self.lr_model)
            ],
            voting='soft',  # Use probability estimates
            weights=ensemble_weights
        )
        
        # Train ensemble (fits on full feature set for simplicity)
        self.ensemble.fit(X_train, y_train)
        
        # Evaluate ensemble
        ensemble_pred = self.ensemble.predict(X_test)
        ensemble_acc = accuracy_score(y_test, ensemble_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(self.ensemble, X, y, cv=5, scoring='accuracy')
        
        print(f"\n  - Ensemble accuracy: {ensemble_acc:.3f}")
        print(f"  - Cross-validation: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
        
        return {
            'ensemble_accuracy': ensemble_acc,
            'xgb_accuracy': xgb_acc,
            'rf_accuracy': rf_acc,
            'lr_accuracy': lr_acc,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': classification_report(
                y_test, ensemble_pred,
                target_names=['Home Win', 'Draw', 'Away Win']
            ),
            'individual_accuracies': {
                'XGBoost': xgb_acc,
                'Random Forest': rf_acc,
                'Logistic Regression': lr_acc
            }
        }
    
    def predict(self, features: pd.DataFrame, return_individual: bool = False) -> pd.DataFrame:
        """
        Predict match outcomes using ensemble
        
        Args:
            features: DataFrame with match features
            return_individual: If True, return individual model predictions too
        
        Returns:
            DataFrame with predictions and probabilities
        """
        if self.ensemble is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Ensure correct features
        X = features[self.feature_columns]
        
        # Get ensemble predictions
        predictions = self.ensemble.predict(X)
        probabilities = self.ensemble.predict_proba(X)
        
        results = pd.DataFrame({
            'prediction': [self.inverse_label_mapping[p] for p in predictions],
            'home_win_prob': probabilities[:, 0],
            'draw_prob': probabilities[:, 1],
            'away_win_prob': probabilities[:, 2],
        })
        
        # Add individual model predictions if requested
        if return_individual:
            xgb_probs = self.xgb_model.predict_proba(X)
            rf_probs = self.rf_model.predict_proba(X)
            
            results['xgb_home_prob'] = xgb_probs[:, 0]
            results['xgb_draw_prob'] = xgb_probs[:, 1]
            results['xgb_away_prob'] = xgb_probs[:, 2]
            
            results['rf_home_prob'] = rf_probs[:, 0]
            results['rf_draw_prob'] = rf_probs[:, 1]
            results['rf_away_prob'] = rf_probs[:, 2]
        
        # Add match info if available
        if 'match_id' in features.columns:
            results['match_id'] = features['match_id'].values
        if 'home_team_id' in features.columns:
            results['home_team_id'] = features['home_team_id'].values
        if 'away_team_id' in features.columns:
            results['away_team_id'] = features['away_team_id'].values
        
        return results
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance from XGBoost and Random Forest
        
        Returns combined importance scores
        """
        if self.xgb_model is None or self.rf_model is None:
            raise ValueError("Models not trained.")
        
        # XGBoost importance
        xgb_importance = self.xgb_model.feature_importances_
        
        # Random Forest importance
        rf_importance = self.rf_model.feature_importances_
        
        # Combine (average)
        combined_importance = (xgb_importance + rf_importance) / 2
        
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'xgb_importance': xgb_importance,
            'rf_importance': rf_importance,
            'combined_importance': combined_importance
        }).sort_values('combined_importance', ascending=False)
        
        return importance_df.head(top_n)
    
    def save_model(self, filepath: str):
        """Save ensemble model"""
        if self.ensemble is None:
            raise ValueError("Model not trained.")
        
        joblib.dump({
            'ensemble': self.ensemble,
            'xgb_model': self.xgb_model,
            'rf_model': self.rf_model,
            'lr_model': self.lr_model,
            'feature_columns': self.feature_columns,
            'label_mapping': self.label_mapping
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load ensemble model"""
        data = joblib.load(filepath)
        self.ensemble = data['ensemble']
        self.xgb_model = data['xgb_model']
        self.rf_model = data['rf_model']
        self.lr_model = data['lr_model']
        self.feature_columns = data['feature_columns']
        self.label_mapping = data['label_mapping']
        self.inverse_label_mapping = {v: k for k, v in self.label_mapping.items()}
