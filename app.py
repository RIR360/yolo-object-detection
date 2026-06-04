import streamlit as st
import cv2
import tempfile
import os
import requests
from PIL import Image
from io import BytesIO
from ultralytics import YOLO

# Set page configuration
st.set_page_config(
    page_title="YOLO Object Detection App",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS to remove default header, reduce max-width, and style title
st.markdown("""
<style>
    /* Remove Streamlit default header, footer, and menu */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .stApp > header {display: none;}
    #MainMenu {visibility: hidden;}
    
    /* Reduce max-width of main content on large screens */
    .main .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 2rem;
        margin: 0 auto;
    }
    
    /* Optional: adjust default padding on very large screens */
    @media (min-width: 1600px) {
        .main .block-container {
            max-width: 1200px;
        }
    }
    
    /* Title styling: big YOLOv8, small subtitle */
    .big-title {
        font-size: 4rem;
        font-weight: 800;
        margin-bottom: 0;
        line-height: 1.2;
    }
    .small-subtitle {
        font-size: 1.5rem;
        font-weight: 400;
        margin-top: 0;
    }
</style>
""", unsafe_allow_html=True)

# Custom title block
st.markdown('<div class="big-title">YOLOv8</div>', unsafe_allow_html=True)
st.markdown('<div class="small-subtitle">Object Detection Tool</div>', unsafe_allow_html=True)

# Load the YOLO model
@st.cache_resource
def load_yolo_model():
    # Use yolov8m.pt from the workspace directory
    return YOLO("yolov8m.pt")

try:
    model = load_yolo_model()
except Exception as e:
    st.error(f"Error loading YOLO model: {e}. Please ensure 'yolov8m.pt' is present in the workspace.")
    model = None

# About this Project section
st.markdown("""
### About this Project
This application is a **YOLOv8 Object Detection Tool** that allows users to perform object detection on images and videos. 
Users can upload an image or paste an image URL, or upload a video (where the tool will automatically process up to the first 30 seconds of footage).

**Creators:** Foysal Ahmed & Ashraf Mia  
*Note: This is a study project only.*
""")

# Create side-by-side layout for Sample Media and Main Detector
col_samples, col_detector = st.columns([1, 2])

with col_samples:
    st.subheader("Sample Detection Media")
    
    # Show Sample Image
    if os.path.exists("image-detection.png"):
        st.write("**Sample Image Detection Results:**")
        st.image("image-detection.png", use_container_width=True)
    else:
        st.warning("Sample image 'image-detection.png' not found.")
        
    # Show Sample Video
    if os.path.exists("video-detection.mp4"):
        st.write("**Sample Video Detection:**")
        st.video("video-detection.mp4")
    else:
        st.warning("Sample video 'video-detection.mp4' not found.")

with col_detector:
    st.subheader("Object Detector")
    
    if model is None:
        st.warning("Model loading failed. Detection features are disabled.")
    else:
        tab_image, tab_video = st.tabs(["Image Detection", "Video Detection"])
        
        # IMAGE DETECTION TAB
        with tab_image:
            st.write("Upload an image file or paste a web link to detect objects.")
            
            image_source = st.radio("Choose Image Source:", ["Upload File", "Paste Image URL"])
            image_to_process = None
            
            if image_source == "Upload File":
                uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])
                if uploaded_image is not None:
                    try:
                        image_to_process = Image.open(uploaded_image)
                    except Exception as e:
                        st.error(f"Error opening image: {e}")
            else:
                image_url = st.text_input("Enter Image URL (e.g., https://example.com/image.jpg):")
                if image_url:
                    try:
                        response = requests.get(image_url, timeout=10)
                        response.raise_for_status()
                        image_to_process = Image.open(BytesIO(response.content))
                    except Exception as e:
                        st.error(f"Error downloading image from URL: {e}")
            
            if image_to_process is not None:
                st.write("**Original Image:**")
                st.image(image_to_process, use_container_width=True)
                
                if st.button("Run Image Detection"):
                    with st.spinner("Processing image..."):
                        try:
                            # Run inference
                            results = model(image_to_process)
                            # Draw detections on image
                            annotated_image_bgr = results[0].plot()
                            # Convert BGR (OpenCV) to RGB
                            annotated_image_rgb = cv2.cvtColor(annotated_image_bgr, cv2.COLOR_BGR2RGB)
                            
                            st.write("**Detection Results:**")
                            st.image(annotated_image_rgb, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error running object detection: {e}")
                            
        # VIDEO DETECTION TAB
        with tab_video:
            st.write("Upload a video to detect objects. Only the first 30 seconds will be processed.")
            uploaded_video = st.file_uploader("Choose a video file...", type=["mp4", "avi", "mov", "mkv"])
            
            if uploaded_video is not None:
                st.video(uploaded_video)
                
                if st.button("Run Video Detection"):
                    # Save uploaded file to temp file to read via OpenCV
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_input:
                        temp_input.write(uploaded_video.read())
                        temp_input_path = temp_input.name
                    
                    # Create temp output file path
                    temp_output_path = tempfile.mktemp(suffix='.mp4')
                    
                    cap = cv2.VideoCapture(temp_input_path)
                    
                    if not cap.isOpened():
                        st.error("Error opening uploaded video file.")
                    else:
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        if fps <= 0 or os.environ.get("STREAMLIT_TESTING"): 
                            fps = 30.0  # Fallback to standard fps if unable to detect
                        
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        
                        # Limit processing to first 30 seconds
                        max_frames = int(fps * 30)
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        frames_to_process = min(max_frames, total_frames)
                        
                        st.info(f"Processing up to {frames_to_process} frames ({min(30, int(total_frames / fps))} seconds at {fps:.2f} FPS).")
                        
                        # Set up VideoWriter
                        # Note: mp4v is widely supported; for direct web playability, H264 is ideal
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))
                        
                        progress_bar = st.progress(0.0)
                        status_text = st.empty()
                        
                        try:
                            frame_count = 0
                            while cap.isOpened() and frame_count < frames_to_process:
                                ret, frame = cap.read()
                                if not ret:
                                    break
                                
                                # Run inference on current frame
                                results = model(frame)
                                annotated_frame = results[0].plot()
                                
                                # Write frame to output video
                                out.write(annotated_frame)
                                
                                frame_count += 1
                                progress = frame_count / frames_to_process
                                progress_bar.progress(progress)
                                status_text.text(f"Processed frame {frame_count}/{frames_to_process}...")
                                
                            cap.release()
                            out.release()
                            
                            status_text.text("Detection completed! Rendering video...")
                            
                            # Read video file bytes to show in streamlit
                            with open(temp_output_path, 'rb') as f:
                                video_bytes = f.read()
                            
                            st.write("**Processed Video (First 30s):**")
                            st.video(video_bytes)
                            
                        except Exception as e:
                            st.error(f"Error during video detection: {e}")
                        finally:
                            # Clean up temporary files
                            if os.path.exists(temp_input_path):
                                os.remove(temp_input_path)
                            if os.path.exists(temp_output_path):
                                os.remove(temp_output_path)