#!/usr/bin/env python3
"""
Aggressive cleanup: Delete all non-essential folders and consolidate to single README
"""

import os
import shutil

def main():
    print("=" * 80)
    print("AGGRESSIVE CLEANUP - MINIMAL PROJECT")
    print("=" * 80)
    print()
    
    # Delete entire folders we don't need
    FOLDERS_TO_DELETE = [
        'prediction',  # 16 experimental scripts - don't need
        'analysis',    # 19 analysis scripts - don't need
        'results',     # 17 old CSV files - don't need
        'archive',     # 5 reference files - don't need
        'docs',        # 7 markdown files - consolidating into README
    ]
    
    print("🗑️  Deleting unnecessary folders...\n")
    deleted_folders = []
    for folder in FOLDERS_TO_DELETE:
        if os.path.exists(folder):
            file_count = len([f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))])
            folder_size = sum(os.path.getsize(os.path.join(folder, f)) 
                            for f in os.listdir(folder) 
                            if os.path.isfile(os.path.join(folder, f))) / 1024  # KB
            
            shutil.rmtree(folder)
            deleted_folders.append((folder, file_count, folder_size))
            print(f"   🗑️  {folder}/ ({file_count} files, {folder_size:.1f} KB)")
    
    # Delete markdown files except README
    print("\n🗑️  Deleting extra markdown files...\n")
    md_files = [f for f in os.listdir('.') if f.endswith('.md') and f != 'README.md']
    deleted_md = []
    for md_file in md_files:
        os.remove(md_file)
        deleted_md.append(md_file)
        print(f"   🗑️  {md_file}")
    
    # Delete cleanup scripts - we don't need them anymore
    print("\n🗑️  Deleting cleanup scripts...\n")
    cleanup_scripts = ['final_cleanup.py', 'organize_files.py', 'cleanup_remaining.py']
    deleted_scripts = []
    for script in cleanup_scripts:
        if os.path.exists(script):
            os.remove(script)
            deleted_scripts.append(script)
            print(f"   🗑️  {script}")
    
    # Delete data_processing folder if empty
    if os.path.exists('data_processing') and not os.listdir('data_processing'):
        os.rmdir('data_processing')
        print(f"   🗑️  data_processing/ (empty)")
    
    print("\n" + "=" * 80)
    print("📋 SUMMARY")
    print("=" * 80)
    
    total_files_deleted = sum(count for _, count, _ in deleted_folders)
    total_files_deleted += len(deleted_md) + len(deleted_scripts)
    
    print(f"\n   Folders deleted: {len(deleted_folders)}")
    for folder, count, size in deleted_folders:
        print(f"      • {folder}/ - {count} files ({size:.1f} KB)")
    
    print(f"\n   Markdown files deleted: {len(deleted_md)}")
    print(f"   Cleanup scripts deleted: {len(deleted_scripts)}")
    print(f"\n   📊 Total files deleted: {total_files_deleted}")
    
    print("\n✅ Aggressive cleanup complete!")
    print("\n📁 MINIMAL PROJECT STRUCTURE:\n")
    
    # Show remaining structure
    root_files = sorted([f for f in os.listdir('.') 
                        if os.path.isfile(f) and not f.startswith('.')])
    
    py_files = [f for f in root_files if f.endswith('.py')]
    csv_files = [f for f in root_files if f.endswith('.csv')]
    other_files = [f for f in root_files if not (f.endswith('.py') or f.endswith('.csv') or f.endswith('.pkl'))]
    
    print("🏠 Root Directory:")
    print(f"   • {len(py_files)} Python scripts (production only)")
    print(f"   • {len(csv_files)} CSV files (essential data)")
    print(f"   • {len(other_files)} config/doc files")
    print(f"\n   Total: {len(root_files)} files")
    
    print("\n📂 Remaining Folders:")
    folders = sorted([f for f in os.listdir('.') 
                     if os.path.isdir(f) and not f.startswith('.') and f != '__pycache__'])
    for folder in folders:
        if folder not in ['23-24stats', '24-25stats', '25-26stats', 'open-data-master', 'src']:
            file_count = len([f for f in os.listdir(folder) 
                            if os.path.isfile(os.path.join(folder, f))])
            print(f"   • {folder}/ - {file_count} files")
    
    print("\n🎯 Production Files:")
    for py_file in sorted(py_files):
        if py_file not in ['aggressive_cleanup.py']:
            print(f"   • {py_file}")

if __name__ == '__main__':
    main()
