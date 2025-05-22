#!/usr/bin/env python3

import sqlite3
import json
import os

def test_database():
    """Test basic database connectivity and data structure"""
    db_path = "abstracts.db"
    
    if not os.path.exists(db_path):
        print(f"Database {db_path} not found")
        return
    
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Test basic table structure
        cursor.execute("SELECT COUNT(*) as total FROM abstracts")
        total = cursor.fetchone()["total"]
        print(f"Total documents: {total}")
        
        # Test sample authors data
        cursor.execute("SELECT authors, title FROM abstracts LIMIT 1")
        sample = cursor.fetchone()
        if sample:
            print(f"Sample title: {sample['title']}")
            print(f"Sample authors raw: {sample['authors']}")
            print(f"Sample authors type: {type(sample['authors'])}")
            
            # Try to parse the JSON
            try:
                authors_parsed = json.loads(sample['authors'])
                print(f"Parsed authors: {authors_parsed}")
                print(f"Parsed type: {type(authors_parsed)}")
                print(f"First author: {authors_parsed[0] if authors_parsed else 'None'}")
            except json.JSONDecodeError as e:
                print(f"JSON parse error: {e}")
        
        # Test FTS5 table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='abstracts_fts'")
        fts_exists = cursor.fetchone()
        print(f"FTS5 table exists: {fts_exists is not None}")
        
        # Test vector table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='vss_abstracts'")
        vss_exists = cursor.fetchone()
        print(f"Vector table exists: {vss_exists is not None}")
        
        conn.close()
        
    except Exception as e:
        print(f"Database test error: {e}")

def test_json_each():
    """Test json_each functionality specifically"""
    db_path = "abstracts.db"
    
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Test json_each with a specific record
        cursor.execute("""
            SELECT authors FROM abstracts 
            WHERE json_valid(authors) AND authors != '[]' 
            LIMIT 1
        """)
        sample = cursor.fetchone()
        
        if sample:
            print(f"\nTesting json_each with: {sample['authors']}")
            
            # Test json_each functionality
            cursor.execute("""
                SELECT value FROM json_each(?)
            """, (sample['authors'],))
            
            authors = cursor.fetchall()
            print(f"Authors from json_each: {[row['value'] for row in authors]}")
        
        conn.close()
        
    except Exception as e:
        print(f"JSON each test error: {e}")

def test_authors_query():
    """Test the authors aggregation query"""
    db_path = "abstracts.db"
    
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Test the authors query
        cursor.execute("""
            SELECT value as author, COUNT(*) as count
            FROM abstracts, json_each(authors)
            WHERE json_valid(authors)
            GROUP BY value
            ORDER BY count DESC
            LIMIT 5
        """)
        
        top_authors = cursor.fetchall()
        print(f"\nTop 5 authors:")
        for author in top_authors:
            print(f"  {author['author']}: {author['count']} publications")
        
        conn.close()
        
    except Exception as e:
        print(f"Authors query test error: {e}")

if __name__ == "__main__":
    print("Testing database structure and APIs...")
    test_database()
    test_json_each()
    test_authors_query()