import cv2
import torch
import os
import tempfile
import streamlit as st
from transformers import DetrForObjectDetection
from transformers import DetrImageProcessor


st.set_page_config(page_icon="📸", page_title="Helmet detection")
st.header("Helmet Detection")

left_col, right_col = st.columns(2)

file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"])

with left_col:
    st.subheader("Video Feed")
    # Placeholder for video frames
    frame_placeholder = st.empty()

with right_col:
    st.subheader("Statistics")
    # Placeholders for real-time statistics
    heading_text = st.empty()
    workers_text = st.empty()
    helmet_text = st.empty()
    percentage_text = st.empty()
    assessment_text = st.empty()


model_name = "facebook/detr-resnet-50"
image_processor = DetrImageProcessor.from_pretrained(model_name)
model = DetrForObjectDetection.from_pretrained(model_name)


def detect_workers_and_helmets(frame):
    inputs = image_processor(images=frame, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([frame.shape[:2]])
    results = image_processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=target_sizes)[0]

    workers = []
    helmets = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        label_name = model.config.id2label[label.item()]
        if label_name == 'person':
            workers.append(box.tolist())
        elif label_name in ['hat', 'helmet']:
            helmets.append(box.tolist())

    return workers, helmets

def check_helmet_usage(workers, helmets):
    workers_with_helmets = 0
    for worker in workers:
        for helmet in helmets:
            if is_helmet_on_worker(worker, helmet):
                workers_with_helmets += 1
                break
    return workers_with_helmets, len(workers)

def is_helmet_on_worker(worker, helmet):
    w_x1, w_y1, w_x2, w_y2 = worker
    h_x1, h_y1, h_x2, h_y2 = helmet
    return (h_y2 > w_y1 and h_y1 < w_y1 and
            h_x1 < (w_x1 + w_x2) / 2 < h_x2)


def process_video(video_path):
    video = cv2.VideoCapture(video_path)
    frame_count = 0
    total_workers = 0
    total_workers_with_helmets = 0

    while True:
        ret, frame = video.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 30 != 0:
            continue

        workers, helmets = detect_workers_and_helmets(frame)
        workers_with_helmets, num_workers = check_helmet_usage(workers, helmets)

        total_workers += num_workers
        total_workers_with_helmets += workers_with_helmets

        # Drawing bounding boxes
        for worker in workers:
            x1, y1, x2, y2 = [int(i) for i in worker]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        for helmet in helmets:
            x1, y1, x2, y2 = [int(i) for i in helmet]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        cv2.putText(frame, f"Workers with helmets: {workers_with_helmets}/{num_workers}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Processed Frame', frame)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        _, frame_encoded = cv2.imencode('.jpg', frame_rgb)
        frame_bytes = frame_encoded.tobytes()
        frame_placeholder.image(frame_bytes, channels="RGB")

        # Press 'q' to quit video playback
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

    return total_workers_with_helmets, total_workers


def process_single_video(uploaded_file):
    filename = uploaded_file.name
    print(f"Processing video: {filename}")

    # Use a temporary file to store the video content for processing
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
        temp_file.write(uploaded_file.read())  # Copy the content of the uploaded file to the temp file
        temp_file_path = temp_file.name

    # Process the video (using the process_video function defined above)
    workers_with_helmets, total_workers = process_video(temp_file_path)

    # Remove the temporary file after processing
    os.unlink(temp_file_path)

    # Calculate and display statistics
    if total_workers > 0:
        percentage = (workers_with_helmets / total_workers) * 100
        heading_text.write(f"\nFinal Statistics for {filename}:")
        workers_text.write(f"Total workers detected: {total_workers}")
        helmet_text.write(f"Workers wearing helmets: {workers_with_helmets}")
        percentage_text.write(f"Percentage of workers wearing helmets: {percentage:.2f}%")

        # Provide a safety assessment based on helmet usage percentage
        if percentage >= 90:
            assessment_text.write("The majority of workers are using helmets properly.")
        elif percentage >= 50:
            assessment_text.write("Many workers are using helmets, but there's room for improvement.")
        else:
            assessment_text.write("Most workers are not using helmets. Immediate safety measures should be taken.")
    else:
        st.write(f"No workers detected in the video {filename}.")

    st.success("Processing complete.")



# file_paths = './factory-workers.mp4' # Replace with your actual video file paths
go = st.button("Go")
if go: 
    process_single_video(file)

