import mysql.connector
import cv2
import numpy as np
from datetime import datetime
import time

# Global connection
cnx = None
last_connection_time = 0

def get_db():
    """Get database connection with retry logic"""
    global cnx, last_connection_time
    
    try:
        # Check if connection exists and is valid
        if cnx is not None:
            try:
                # Test connection with a simple query
                cursor = cnx.cursor()
                cursor.execute("SELECT 1")
                cursor.fetchall()  # Consume all results
                cursor.close()
                return cnx
            except:
                # Connection is dead, create new one
                try:
                    if cnx:
                        cnx.close()
                except:
                    pass
                cnx = None
        
        # Create new connection
        cnx = mysql.connector.connect(
            host="localhost",
            user="root",
            port=3310,
            password="",
            database="quizo",
            connection_timeout=10,
            autocommit=False,
            buffered=True  # Important: This helps with unread results
        )
        print("✅ Database connected successfully!")
        return cnx
        
    except mysql.connector.Error as err:
        print(f"❌ Database connection error: {err}")
        cnx = None
        return None

def close_cursor(cursor):
    """Safely close cursor and consume any remaining results"""
    try:
        if cursor:
            # Consume any unread results
            while cursor.nextset():
                pass
            cursor.close()
    except:
        pass

def create_proctoring_session(user_email):
    """Create a new proctoring session"""
    db = get_db()
    if not db:
        print("⚠️ No database connection, using session ID 1")
        return 1
        
    cursor = None
    try:
        cursor = db.cursor()
        query = "INSERT INTO proctoring_sessions (user_email, status) VALUES (%s, 'active')"
        cursor.execute(query, (user_email,))
        db.commit()
        session_id = cursor.lastrowid
        print(f"✅ Session created: {session_id} for {user_email}")
        return session_id
    except mysql.connector.Error as err:
        print(f"❌ Error creating session: {err}")
        return 1
    finally:
        close_cursor(cursor)

def insert_proctoring_frame(session_id, timestamp, frame, face_data):
    """Insert frame with face_data dictionary"""
    db = get_db()
    if not db:
        print("⚠️ No database connection, frame not saved")
        return None
        
    cursor = None
    try:
        cursor = db.cursor()
        
        # Convert image to binary
        if frame is None or frame.size == 0:
            print("❌ Invalid frame data")
            return None
            
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
        success, buffer = cv2.imencode('.jpg', frame, encode_params)
        
        if not success:
            print("❌ Failed to encode image")
            return None
            
        image_binary = buffer.tobytes()
        
        # Extract data from dictionary or boolean
        if isinstance(face_data, dict):
            face_detected = face_data.get('face_detected', False)
            face_count = face_data.get('face_count', 1 if face_detected else 0)
            head_pose = face_data.get('head_pose', 'normal')
        else:
            face_detected = bool(face_data)
            face_count = 1 if face_detected else 0
            head_pose = 'normal'
        
        # Ensure values are correct types
        face_detected = bool(face_detected)
        face_count = int(face_count)
        session_id = int(session_id) if session_id else 1
        
        query = """INSERT INTO proctoring_frames 
                  (session_id, timestamp, frame_image, face_detected, face_count, head_pose) 
                  VALUES (%s, %s, %s, %s, %s, %s)"""
        
        cursor.execute(query, (
            session_id, 
            timestamp, 
            image_binary, 
            face_detected, 
            face_count, 
            str(head_pose)[:50]
        ))
        
        db.commit()
        frame_id = cursor.lastrowid
        
        print(f"✅ Frame {frame_id} inserted (Size: {len(image_binary)} bytes)")
        return frame_id
        
    except mysql.connector.Error as err:
        print(f"❌ Database error inserting frame: {err}")
        return None
    except Exception as e:
        print(f"❌ Error inserting frame: {e}")
        return None
    finally:
        close_cursor(cursor)

def insert_suspicious_activity(session_id, frame_id, activity_type, description):
    """Log suspicious activity to database"""
    db = get_db()
    if not db:
        print("⚠️ No database connection, activity not logged")
        return False
    
    cursor = None
    try:
        cursor = db.cursor()
        
        session_id = int(session_id) if session_id else 1
        frame_id = int(frame_id) if frame_id else None
        
        query = """INSERT INTO suspicious_activities 
                  (session_id, frame_id, activity_type, description) 
                  VALUES (%s, %s, %s, %s)"""
        
        cursor.execute(query, (
            session_id, 
            frame_id, 
            str(activity_type)[:100],
            str(description)[:255]
        ))
        
        db.commit()
        print(f"⚠️ Suspicious activity logged: {activity_type}")
        return True
        
    except mysql.connector.Error as err:
        print(f"❌ Database error logging activity: {err}")
        return False
    except Exception as e:
        print(f"❌ Error logging activity: {e}")
        return False
    finally:
        close_cursor(cursor)

def get_suspicious_activities(session_id=None):
    """Get suspicious activities for reporting"""
    db = get_db()
    if not db:
        return []
    
    cursor = None
    try:
        cursor = db.cursor(dictionary=True)
        
        if session_id:
            query = """
                SELECT * FROM suspicious_activities 
                WHERE session_id = %s 
                ORDER BY created_at DESC 
                LIMIT 100
            """
            cursor.execute(query, (session_id,))
        else:
            query = """
                SELECT * FROM suspicious_activities 
                ORDER BY created_at DESC 
                LIMIT 100
            """
            cursor.execute(query)
            
        activities = cursor.fetchall()
        
        # Format datetime objects to strings
        for activity in activities:
            if 'created_at' in activity and activity['created_at']:
                activity['created_at'] = activity['created_at'].strftime('%Y-%m-%d %H:%M:%S')
        
        print(f"✅ Retrieved {len(activities)} suspicious activities")
        return activities
        
    except mysql.connector.Error as err:
        print(f"❌ Database error fetching activities: {err}")
        return []
    except Exception as e:
        print(f"❌ Error fetching activities: {e}")
        return []
    finally:
        close_cursor(cursor)

def get_latest_frames(limit=20):
    """Get latest frames"""
    db = get_db()
    if not db:
        return []
        
    cursor = None
    try:
        cursor = db.cursor(dictionary=True)
        
        query = """
            SELECT id, session_id, timestamp, head_pose, face_detected, face_count
            FROM proctoring_frames 
            ORDER BY id DESC 
            LIMIT %s
        """
        
        cursor.execute(query, (limit,))
        frames = cursor.fetchall()
        
        # Format data for JSON
        for frame in frames:
            if frame['timestamp']:
                frame['timestamp'] = str(frame['timestamp'])
            frame['face_detected'] = bool(frame['face_detected'])
            frame['face_count'] = int(frame['face_count']) if frame['face_count'] else 0
            
        print(f"✅ Retrieved {len(frames)} frames")
        return frames
        
    except mysql.connector.Error as err:
        print(f"❌ Database error: {err}")
        return []
    except Exception as e:
        print(f"❌ Error: {e}")
        return []
    finally:
        close_cursor(cursor)

def get_stats():
    """Get dashboard statistics"""
    db = get_db()
    if not db:
        return {"total_frames": 0, "suspicious_count": 0, "session_count": 0, "face_count": 0}
        
    cursor = None
    try:
        cursor = db.cursor(dictionary=True)
        
        cursor.execute("SELECT COUNT(*) as count FROM proctoring_frames")
        result = cursor.fetchone()
        total_frames = result['count'] if result else 0
        
        cursor.execute("SELECT COUNT(*) as count FROM suspicious_activities")
        result = cursor.fetchone()
        suspicious_count = result['count'] if result else 0
        
        cursor.execute("SELECT COUNT(*) as count FROM proctoring_sessions")
        result = cursor.fetchone()
        session_count = result['count'] if result else 0
        
        cursor.execute("SELECT COUNT(*) as count FROM proctoring_frames WHERE face_detected = 1")
        result = cursor.fetchone()
        face_count = result['count'] if result else 0
        
        stats = {
            "total_frames": int(total_frames),
            "suspicious_count": int(suspicious_count),
            "session_count": int(session_count),
            "face_count": int(face_count)
        }
        
        print(f"📊 Stats: {stats}")
        return stats
        
    except mysql.connector.Error as err:
        print(f"❌ Database error in stats: {err}")
        return {"total_frames": 0, "suspicious_count": 0, "session_count": 0, "face_count": 0}
    except Exception as e:
        print(f"❌ Stats error: {e}")
        return {"total_frames": 0, "suspicious_count": 0, "session_count": 0, "face_count": 0}
    finally:
        close_cursor(cursor)

def get_frame_image(frame_id):
    """Get frame image by ID"""
    db = get_db()
    if not db:
        return None
        
    cursor = None
    try:
        cursor = db.cursor()
        cursor.execute("SELECT frame_image FROM proctoring_frames WHERE id = %s", (frame_id,))
        result = cursor.fetchone()
        
        if result and result[0]:
            return result[0]
        return None
        
    except mysql.connector.Error as err:
        print(f"❌ Error getting frame: {err}")
        return None
    finally:
        close_cursor(cursor)

def end_proctoring_session(session_id):
    """End a proctoring session"""
    db = get_db()
    if not db:
        return False
        
    cursor = None
    try:
        cursor = db.cursor()
        query = "UPDATE proctoring_sessions SET end_time = NOW(), status = 'ended' WHERE id = %s"
        cursor.execute(query, (session_id,))
        db.commit()
        print(f"✅ Session {session_id} ended")
        return True
        
    except mysql.connector.Error as err:
        print(f"❌ Error ending session: {err}")
        return False
    finally:
        close_cursor(cursor)
    
    
    

def create_tables():
    """Create necessary tables if they don't exist"""
    db = get_db()
    if not db:
        return False
        
    cursor = None
    try:
        cursor = db.cursor()
        
        # Create sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS proctoring_sessions (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_email VARCHAR(100),
                start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                end_time TIMESTAMP NULL,
                status VARCHAR(50) DEFAULT 'active'
            )
        """)
        
        # Create frames table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS proctoring_frames (
                id INT AUTO_INCREMENT PRIMARY KEY,
                session_id INT,
                timestamp VARCHAR(50),
                frame_image LONGBLOB,
                face_detected BOOLEAN DEFAULT FALSE,
                face_count INT DEFAULT 0,
                head_pose VARCHAR(50),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX (session_id),
                INDEX (created_at)
            )
        """)
        
        # Create suspicious activities table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS suspicious_activities (
                id INT AUTO_INCREMENT PRIMARY KEY,
                session_id INT,
                frame_id INT,
                activity_type VARCHAR(100),
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX (session_id),
                INDEX (created_at)
            )
        """)
        
        db.commit()
        print("✅ Tables created/verified successfully")
        return True
        
    except mysql.connector.Error as err:
        print(f"❌ Error creating tables: {err}")
        return False
    finally:
        close_cursor(cursor)

def close_all_connections():
    """Close all database connections"""
    global cnx
    try:
        if cnx and cnx.is_connected():
            cnx.close()
            cnx = None
            print("✅ Database connection closed")
    except:
        pass

# Test function
if __name__ == "__main__":
    print("="*50)
    print("DATABASE CONNECTION TEST")
    print("="*50)
    
    # Create tables first
    create_tables()
    
    # Test connection
    db = get_db()
    if db:
        print("✅ Database connected")
        
        # Test session creation
        session_id = create_proctoring_session("test@test.com")
        if session_id:
            print(f"✅ Session created: {session_id}")
            
            # Create test frame
            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(dummy_frame, "TEST FRAME", (50, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Test frame insertion
            face_data = {'face_detected': True, 'face_count': 1, 'head_pose': 'normal'}
            frame_id = insert_proctoring_frame(session_id, "12:00:00", dummy_frame, face_data)
            
            if frame_id:
                print(f"✅ Test frame inserted: {frame_id}")
                
                # Test suspicious activity
                insert_suspicious_activity(session_id, frame_id, "test_activity", "Test description")
                
                # Test retrieval
                frames = get_latest_frames(5)
                print(f"✅ Retrieved {len(frames)} frames")
                
                activities = get_suspicious_activities()
                print(f"✅ Retrieved {len(activities)} activities")
                
                stats = get_stats()
                print(f"✅ Stats: {stats}")
            else:
                print("❌ Frame insertion failed")
        else:
            print("❌ Session creation failed")
    else:
        print("❌ Database connection failed")
        
def insert_suspicious_activity(session_id, frame_id, activity_type, description):
    """Log suspicious activity to database - THREAD SAFE version"""
    # Har baar naya connection banao, global connection use mat karo
    try:
        # Create NEW connection (not global)
        cnx = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="quizo",
            connection_timeout=5
        )
        
        cursor = cnx.cursor()
        
        session_id = int(session_id) if session_id else 1
        frame_id = int(frame_id) if frame_id else None
        
        query = """INSERT INTO suspicious_activities 
                  (session_id, frame_id, activity_type, description) 
                  VALUES (%s, %s, %s, %s)"""
        
        cursor.execute(query, (
            session_id, 
            frame_id, 
            str(activity_type)[:100],
            str(description)[:255]
        ))
        
        cnx.commit()
        cursor.close()
        cnx.close()
        print(f"⚠️ Suspicious activity logged: {activity_type}")
        return True
        
    except Exception as e:
        print(f"❌ Error logging activity: {e}")
        return False
    
    # Close connection
    close_all_connections()
    print("="*50)