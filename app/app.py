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
        overflow-x: hidden; /* Prevent horizontal overflow */
        box-sizing: border-box;
    }

    /* Overlay for better readability */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.5);
        z-index: -1;
        box-sizing: border-box;
    }

    /* Main container */
    .main {
        width: 100%;
        box-sizing: border-box;
    }

    /* Individual section cards with updated blue background and white text */
    .section-card {
        background: rgba(60, 120, 190, 0.95);
        border-radius: 5px;
        padding: 9px; /* As previously adjusted */
        margin-bottom: 2px;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        color: #ffffff !important;
        overflow: auto;
        width: 100%;
        box-sizing: border-box;
    }

    .section-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
    }

    /* Ensure all text inside section-card is white and visible */
    .section-card p, .section-card div, .section-card span, .section-card .stMarkdown, .section-card .stWrite, .section-card .stWarning, .section-card .stError {
        color: #ffffff !important;
        text-shadow: 0 0 1px rgba(0, 0, 0, 0.3);
    }

    /* Header styling (simplified) */
    .stHeader {
        color: #ffffff;
        font-family: 'Roboto', sans-serif;
        font-size: 2em;
        font-weight: 500;
        text-align: center;
        padding: 5px;
        background: rgba(30, 58, 138, 0.8);
        border-radius: 5px;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.1);
        margin: 25px auto 2px auto; /* Add top margin for spacing */
        width: 100%;
        box-sizing: border-box;
    }

    /* Subheader styling */
    .stSubheader {
        color: #ffffff;
        font-family: 'Roboto', sans-serif;
        font-size: 1.2em;
        margin-top: 2px;
        margin-bottom: 2px;
        text-shadow: none;
        animation: fadeIn 1s ease-in;
    }

    /* Warning styling */
    .stWarning {
        background-color: rgba(254, 242, 242, 0.9);
        padding: 5px;
        border-radius: 3px;
        border-left: 3px solid #dc2626;
        animation: slideIn 0.8s ease-out;
        color: #ffffff;
        margin-bottom: 2px;
    }

    /* Error styling */
    .stError {
        background-color: rgba(254, 242, 242, 0.9);
        padding: 5px;
        border-radius: 3px;
        border-left: 3px solid #dc2626;
        animation: slideIn 0.8s ease-out;
        color: #ffffff;
        margin-bottom: 2px;
    }

    /* Success message styling */
    .stSuccess {
        background-color: rgba(255, 255, 255, 0.9);
        color: #333333;
        padding: 5px;
        border-radius: 3px;
        margin-top: 2px;
        margin-bottom: 2px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
        animation: slideIn 0.8s ease-out;
    }

    /* Processing message styling */
    .processing-message {
        background-color: rgba(255, 255, 255, 0.9);
        color: #000000;
        padding: 5px;
        border-radius: 3px;
        display: inline-block;
        margin-bottom: 2px;
    }

    /* Button styling */
    .stButton>button {
        background: linear-gradient(45deg, #28a745, #218838);
        color: #ffffff;
        border: none;
        padding: 5px 10px;
        border-radius: 3px;
        font-size: 0.9em;
        font-family: 'Roboto', sans-serif;
        transition: transform 0.3s ease, background 0.3s ease;
        position: relative;
        overflow: hidden;
        margin-bottom: 2px;
    }

    .stButton>button:hover {
        background: linear-gradient(45deg, #218838, #1e7e34);
        transform: translateY(-2px);
    }

    .stButton>button::before {
        content: '▶';
        position: absolute;
        left: -15px;
        opacity: 0;
        transition: all 0.3s ease;
    }

    .stButton>button:hover::before {
        left: 5px;
        opacity: 1;
    }

    /* Selectbox and file uploader styling */
    .stSelectbox div, .stFileUploader {
        background-color: rgba(255, 255, 255, 0.15);
        border-radius: 3px;
        padding: 5px;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
        color: #ffffff !important;
        margin-bottom: 2px;
    }

    .stSelectbox div:hover, .stFileUploader:hover {
        transform: scale(1.01);
    }

    /* Dataframe styling */
    .stDataFrame {
        background-color: rgba(255, 255, 255, 0.15);
        border-radius: 3px;
        padding: 5px;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.1);
        animation: fadeIn 1s ease-in;
        color: #ffffff !important;
        margin-bottom: 2px;
    }

    /* Animations */
    @keyframes floatUp {
        from { transform: translateY(10px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }

    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }

    @keyframes slideIn {
        from { transform: translateX(-10px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load Google Font (Updated to Roboto)
st.markdown(
    '<link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap" rel="stylesheet">',
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
    # Decorative header (spanning full width)
    st.markdown('<div class="stHeader">Object Detection and Tracking</div>', unsafe_allow_html=True)

    # All sections in a single column
    # Upload Video
    with st.container():
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="stSubheader">Upload Video</div>', unsafe_allow_html=True)
        video_file = st.file_uploader("Upload a video", type=["mp4", "avi"])
        st.markdown('</div>', unsafe_allow_html=True)

    # Select Objects to Detect
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

        # Real-Time Detections
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

        # Filter Detections
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

        # Evaluation Metrics (adapted from classmates' approach)
        with st.container():
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="stSubheader">Evaluation Metrics</div>', unsafe_allow_html=True)

            # Check if there are any detections
            if not log_data:
                st.markdown(
                    '<p style="color: #000000; background-color: #ffffff; padding: 5px; border-radius: 3px; margin-bottom: 2px;">No ground truth provided, returning zero metrics for detection evaluation.</p>',
                    unsafe_allow_html=True
                )
                st.markdown(
                    '<p style="color: #000000; background-color: #ffffff; padding: 5px; border-radius: 3px; margin-bottom: 2px;">No ground truth provided, returning zero metrics for tracking evaluation.</p>',
                    unsafe_allow_html=True
                )
            else:
                # Detection Evaluation (adapted without ground truth)
                st.markdown(
                    '<p style="color: #000000; background-color: #ffffff; padding: 5px; border-radius: 3px; margin-bottom: 2px;"><strong>Detection Evaluation</strong></p>',
                    unsafe_allow_html=True
                )
                # Since we don't have ground truth, we can't compute TP, FP, FN
                precision = 0.0
                recall = 0.0
                mAP = 0.0
                st.markdown(
                    f'<p style="color: #000000; background-color: #ffffff; padding: 5px; border-radius: 3px; margin-bottom: 2px;">Precision: {precision:.2f} (Requires ground truth)</p>',
                    unsafe_allow_html=True
                )
                st.markdown(
                    f'<p style="color: #000000; background-color: #ffffff; padding: 5px; border-radius: 3px; margin-bottom: 2px;">Recall: {recall:.2f} (Requires ground truth)</p>',
                    unsafe_allow_html=True
                )
                st.markdown(
                    f'<p style="color: #000000; background-color: #ffffff; padding: 5px; border-radius: 3px; margin-bottom: 2px;">mAP: {mAP:.2f} (Requires ground truth)</p>',
                    unsafe_allow_html=True
                )

                # We can still show confidence-based metrics
                confidences = [log["confidence"] for log in log_data]
                total_detections = len(confidences)
                avg_confidence = np.mean(confidences) if confidences else 0
                reliable_detections = len([c for c in confidences if c >= 0.5])
                reliable_percentage = (reliable_detections / total_detections * 100) if total_detections > 0 else 0

                st.markdown(
                    f'<p style="color: #000000; background-color: #ffffff; padding: 5px; border-radius: 3px; margin-bottom: 2px;">Total Detections: {total_detections}</p>',
                    unsafe_allow_html=True
                )
                st.markdown(
                    f'<p style="color: #000000; background-color: #ffffff; padding: 5px; border-radius: 3px; margin-bottom: 2px;">Average Confidence Score: {avg_confidence:.2f}</p>',
                    unsafe_allow_html=True
                )
                st.markdown(
                    f'<p style="color: #000000; background-color: #ffffff; padding: 5px; border-radius: 3px; margin-bottom: 5px;">Reliable Detections (Confidence ≥ 0.5): {reliable_detections} ({reliable_percentage:.2f}%)</p>',
                    unsafe_allow_html=True
                )

                # Tracking Evaluation (adapted without ground truth)
                st.markdown(
                    '<p style="color: #000000; background-color: #ffffff; padding: 5px; border-radius: 3px; margin-bottom: 2px;"><strong>Tracking Evaluation</strong></p>',
                    unsafe_allow_html=True
                )
                id_switches = count_id_switches(log_data)
                mota = 0.0  # Cannot compute without ground truth
                st.markdown(
                    f'<p style="color: #000000; background-color: #ffffff; padding: 5px; border-radius: 3px; margin-bottom: 2px;">ID Switches: {id_switches}</p>',
                    unsafe_allow_html=True
                )
                st.markdown(
                    f'<p style="color: #000000; background-color: #ffffff; padding: 5px; border-radius: 3px; margin-bottom: 2px;">MOTA: {mota:.2f} (Requires ground truth)</p>',
                    unsafe_allow_html=True
                )

            st.markdown('</div>', unsafe_allow_html=True)

        # Logs
        with st.container():
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="stSubheader">Logs</div>', unsafe_allow_html=True)
            if log_data:
                save_log(log_data, "tracking_log_web.csv")
                st.download_button("Download Log", data=open("tracking_log_web.csv", "rb"), file_name="tracking_log_web.csv")
            st.markdown('</div>', unsafe_allow_html=True)

        # Cleanup: Release video capture and remove temporary file
        cap.release()
        os.remove(tfile.name)

if __name__ == "__main__":
    main()
