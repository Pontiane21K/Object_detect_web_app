import streamlit as st
import cv2
import tempfile
import os
import sys
import pandas as pd
import time
import numpy as np

# Custom CSS directly embedded
st.markdown(
    """
    <style>
    /* App background with tech-themed image */
    .stApp {
        background: url('https://images.unsplash.com/photo-1501854140801-50d01698950b?auto=format&fit=crop&w=1350&q=80') no-repeat center center fixed;
        background-size: cover;
    }

    /* Overlay for better readability */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.4); /* Dark overlay to match blue/green theme */
        z-index: -1;
    }

    /* Main container */
    .main {
        padding: 20px;
        z-index: 1;
    }

    /* Individual section cards with blue background and white text */
    .section-card {
        background: rgba(43, 108, 176, 0.9); /* Blue background matching theme */
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        color: #ffffff !important; /* White text */
    }

    .section-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 25px rgba(0, 0, 0, 0.4);
    }

    /* Ensure all text inside section-card is white */
    .section-card p, .section-card div, .section-card span, .section-card .stMarkdown, .section-card .stWrite, .section-card .stWarning, .section-card .stError {
        color: #ffffff !important;
    }

    /* Specific style for Metrics section with white background */
    .metrics-section {
        background: #ffffff; /* White background for better readability */
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    .metrics-section:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 25px rgba(0, 0, 0, 0.4);
    }

    /* Ensure all text inside metrics-section is black */
    .metrics-section p, .metrics-section div, .metrics-section span, .metrics-section .stMarkdown, .metrics-section .stWrite, .metrics-section .stWarning, .metrics-section .stError {
        color: #000000 !important;
    }

    /* Ensure buttons inside metrics-section are readable */
    .metrics-section .stDownloadButton>button {
        color: #ffffff !important; /* White text on blue gradient */
        background: linear-gradient(45deg, #007bff, #0056b3);
    }

    .metrics-section .stDownloadButton>button:hover {
        background: linear-gradient(45deg, #0056b3, #003d7a);
    }

    /* Header styling without separation bar */
    .stHeader {
        color: #f0e8e0;
        font-family: 'Playfair Display', serif;
        font-size: 2.8em;
        font-weight: 700;
        text-align: center;
        padding: 15px;
        background: rgba(30, 58, 138, 0.9); /* Dark blue header */
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        position: relative;
        overflow: hidden;
        margin-bottom: 30px;
    }

    /* Subheader styling without separation bar */
    .stSubheader {
        color: #ffffff; /* White text */
        font-family: 'Playfair Display', serif;
        font-size: 1.7em;
        margin-top: 15px;
        text-shadow: 1px 1px 4px rgba(0, 0, 0, 0.3);
        animation: fadeIn 1s ease-in;
    }

    /* Subheader inside metrics section */
    .metrics-section .stSubheader {
        color: #000000; /* Black text to match the white background */
        text-shadow: none; /* Remove shadow for better readability */
    }

    /* Warning styling */
    .stWarning {
        background-color: rgba(254, 242, 242, 0.9);
        padding: 12px;
        border-radius: 8px;
        border-left: 5px solid #dc2626;
        animation: slideIn 0.8s ease-out;
        color: #ffffff; /* White text for contrast */
    }

    /* Error styling */
    .stError {
        background-color: rgba(254, 242, 242, 0.9);
        padding: 12px;
        border-radius: 8px;
        border-left: 5px solid #dc2626;
        animation: slideIn 0.8s ease-out;
        color: #ffffff; /* White text for contrast */
    }

    /* Success message styling with white background */
    .stSuccess {
        background-color: #ffffff; /* White background */
        color: #333333; /* Dark gray text for contrast */
        padding: 10px 15px;
        border-radius: 8px;
        margin-top: 10px;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        animation: slideIn 0.8s ease-out;
    }

    /* Processing message styling */
    .processing-message {
        background-color: #ffffff; /* White background */
        color: #000000; /* Black text */
        padding: 10px 15px;
        border-radius: 8px;
        display: inline-block;
    }

    /* Button styling */
    .stButton>button {
        background: linear-gradient(45deg, #28a745, #218838); /* Green gradient */
        color: #ffffff; /* White text */
        border: none;
        padding: 12px 25px;
        border-radius: 8px;
        font-size: 1.1em;
        transition: transform 0.3s ease, background 0.3s ease;
        position: relative;
        overflow: hidden;
    }

    .stButton>button:hover {
        background: linear-gradient(45deg, #218838, #1e7e34);
        transform: translateY(-2px);
    }

    .stButton>button::before {
        content: '▶';
        position: absolute;
        left: -20px;
        opacity: 0;
        transition: all 0.3s ease;
    }

    .stButton>button:hover::before {
        left: 10px;
        opacity: 1;
    }

    /* Selectbox and file uploader styling */
    .stSelectbox div, .stFileUploader {
        background-color: rgba(255, 255, 255, 0.1); /* Lighter blue tint */
        border-radius: 8px;
        padding: 10px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
        transition: transform 0.3s ease;
        color: #ffffff !important; /* White text */
    }

    .stSelectbox div:hover, .stFileUploader:hover {
        transform: scale(1.02);
    }

    /* Dataframe styling */
    .stDataFrame {
        background-color: rgba(255, 255, 255, 0.1); /* Lighter blue tint */
        border-radius: 8px;
        padding: 10px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
        animation: fadeIn 1s ease-in;
        color: #ffffff !important; /* White text */
    }

    /* Dataframe inside metrics section */
    .metrics-section .stDataFrame {
        background-color: rgba(0, 0, 0, 0.05); /* Slight gray tint for contrast */
        color: #000000 !important; /* Black text */
    }

    /* Animations */
    @keyframes floatUp {
        from { transform: translateY(20px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }

    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }

    @keyframes slideIn {
        from { transform: translateX(-20px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load Google Font
st.markdown(
    '<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&display=swap" rel="stylesheet">',
    unsafe_allow_html=True
)

# Custom function to display a progress bar
def show_progress(status_text):
    progress_bar = st.progress(0)
    for i in range(100):
        time.sleep(0.01)  # Simulate processing
        progress_bar.progress(i + 1)
    st.success(status_text)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.yolo_video import load_video, load_model, process_frame, save_log, count_id_switches

def main():
    # Decorative header
    st.markdown('<div class="stHeader">Object Detection and Tracking</div>', unsafe_allow_html=True)
    
    # User interface with separated sections
    with st.container():
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="stSubheader">Upload Video</div>', unsafe_allow_html=True)
        video_file = st.file_uploader("Upload a video", type=["mp4", "avi"])
        st.markdown('</div>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="stSubheader">Select Objects to Detect</div>', unsafe_allow_html=True)
        col1, col2 = st.columns([3, 1])
        with col1:
            available_classes = list(load_model("model/yolov8n.pt")[1].values())
            selected_classes = st.multiselect(
                "Choose classes to detect",
                options=available_classes,
                default=["person", "car", "traffic light"],
                help="Select the objects you want to detect in the video."
            )
        with col2:
            st.image("https://via.placeholder.com/100x100.png?text=Logo", use_container_width=True)  # Placeholder for logo
        st.markdown('</div>', unsafe_allow_html=True)

    # Check if selected classes are valid
    invalid_classes = [cls for cls in selected_classes if cls not in available_classes]
    if invalid_classes:
        st.error(f"The following objects are not supported: {', '.join(invalid_classes)}. Please choose from the available options.")
        return
    
    selected_class_ids = [k for k, v in load_model("model/yolov8n.pt")[1].items() if v in selected_classes]
    
    if not selected_classes:
        st.warning("Please select at least one class for detection.")
        return

    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(video_file.read())
        
        cap = load_video(tfile.name)
        stframe = st.empty()
        status = st.empty()
        status.markdown('<div class="processing-message">Processing video...</div>', unsafe_allow_html=True)
        show_progress("Video processed successfully!")
        
        with st.container():
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="stSubheader">Real-Time Detections</div>', unsafe_allow_html=True)
            table_placeholder = st.empty()
            log_data = []
            table_data = []
            objects_detected = False
            frame_count = 0
            detected_classes = set()
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1

                # Pass an empty list for ground truth since we don't need it
                frame, frame_log, detected, _ = process_frame(
                    frame, load_model("model/yolov8n.pt")[0], load_model("model/yolov8n.pt")[1], selected_class_ids=selected_class_ids,
                    ground_truth=[],  # No ground truth needed
                    frame_number=frame_count
                )
                log_data.extend(frame_log)
                if detected:
                    objects_detected = True
                    for log in frame_log:
                        detected_classes.add(log["class"])
                        table_data.append({
                            "Alert": f"Alert: {log['class']} detected",
                            "Object ID": log["track_id"],
                            "Class": log["class"],
                            "X1": round(log["x1"], 2),
                            "Y1": round(log["y1"], 2),
                            "X2": round(log["x2"], 2),
                            "Y2": round(log["y2"], 2),
                            "Time": log["timestamp"],
                            "Confidence": round(log["confidence"], 2)
                        })
                    table_df = pd.DataFrame(table_data)
                    table_placeholder.dataframe(table_df, use_container_width=True)
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                stframe.image(frame_rgb, use_container_width=True)
            
            # Display detection table with message for undetected classes
            table_df = pd.DataFrame(table_data)
            table_placeholder.dataframe(table_df, use_container_width=True)
            non_detected_classes = [cls for cls in selected_classes if cls not in detected_classes]
            if non_detected_classes:
                st.warning(f"The following objects were not detected: {', '.join(non_detected_classes)}.")
            if not objects_detected:
                st.warning("No objects detected in the video.")
            st.markdown('</div>', unsafe_allow_html=True)

        with st.container():
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="stSubheader">Filter Detections</div>', unsafe_allow_html=True)
            filter_class = st.selectbox("Filter by class", options=["All"] + selected_classes, index=0)
            if filter_class != "All":
                filtered_data = [d for d in table_data if d["Class"] == filter_class]
            else:
                filtered_data = table_data
            filtered_df = pd.DataFrame(filtered_data)
            st.dataframe(filtered_df, use_container_width=True)
            if filtered_data:
                csv = filtered_df.to_csv(index=False)
                st.download_button("Download Table", data=csv, file_name="detection_table.csv")
            st.markdown('</div>', unsafe_allow_html=True)

       with st.container():
                st.markdown('<div class="section-card">', unsafe_allow_html=True)
                st.markdown('<div class="stSubheader">Evaluation Metrics</div>', unsafe_allow_html=True)
                # Debug: Check if log_data is populated
                if not log_data:
                    st.markdown(
                        '<p style="color: #ffffff; background-color: rgba(30, 58, 138, 0.9); padding: 8px; border-radius: 5px;">Debug: log_data is empty. No detections were recorded.</p>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<p style="color: #ffffff; background-color: rgba(30, 58, 138, 0.9); padding: 8px; border-radius: 5px;">Debug: log_data contains {len(log_data)} entries.</p>',
                        unsafe_allow_html=True
                    )
                    # Detection Metrics (approximate based on confidence scores)
                    st.markdown(
                        '<p style="color: #ffffff; background-color: rgba(30, 58, 138, 0.9); padding: 8px; border-radius: 5px;"><strong>Detection Quality (Approximate)</strong></p>',
                        unsafe_allow_html=True
                    )
                    confidences = [log["confidence"] for log in log_data]
                    total_detections = len(confidences)
                    avg_confidence = np.mean(confidences) if confidences else 0
                    reliable_detections = len([c for c in confidences if c >= 0.5])
                    reliable_percentage = (reliable_detections / total_detections * 100) if total_detections > 0 else 0

                    st.markdown(
                        f'<p style="color: #ffffff; background-color: rgba(30, 58, 138, 0.9); padding: 8px; border-radius: 5px;">Total Detections: {total_detections}</p>',
                        unsafe_allow_html=True
                    )
                    st.markdown(
                        f'<p style="color: #ffffff; background-color: rgba(30, 58, 138, 0.9); padding: 8px; border-radius: 5px;">Average Confidence Score: {avg_confidence:.2f}</p>',
                        unsafe_allow_html=True
                    )
                    st.markdown(
                        f'<p style="color: #ffffff; background-color: rgba(30, 58, 138, 0.9); padding: 8px; border-radius: 5px;">Reliable Detections (Confidence ≥ 0.5): {reliable_detections} ({reliable_percentage:.2f}%)</p>',
                        unsafe_allow_html=True
                    )
                    st.markdown(
                        '<p style="color: #ffffff; background-color: rgba(30, 58, 138, 0.9); padding: 8px; border-radius: 5px;">Note: True Precision, Recall, and mAP cannot be calculated without ground truth annotations.</p>',
                        unsafe_allow_html=True
                    )

                    # Tracking Metrics
                    st.markdown(
                        '<p style="color: #ffffff; background-color: rgba(30, 58, 138, 0.9); padding: 8px; border-radius: 5px;"><strong>Tracking Performance</strong></p>',
                        unsafe_allow_html=True
                    )
                    id_switches = count_id_switches(log_data)
                    st.markdown(
                        f'<p style="color: #ffffff; background-color: rgba(30, 58, 138, 0.9); padding: 8px; border-radius: 5px;">ID Switches: {id_switches}</p>',
                        unsafe_allow_html=True
                    )
                    st.markdown(
                        '<p style="color: #ffffff; background-color: rgba(30, 58, 138, 0.9); padding: 8px; border-radius: 5px;">MOTA (Multiple Object Tracking Accuracy) cannot be calculated: no ground truth available.</p>',
                        unsafe_allow_html=True
                    )
                st.markdown('</div>', unsafe_allow_html=True)

    
        cap.release()
        os.remove(tfile.name)

if __name__ == "__main__":
    main()
