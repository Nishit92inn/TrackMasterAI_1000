# Face Detection Project

## Project Summary
This project involves developing a web application that allows users to scrape images of celebrities, process these images to detect and extract faces, and display the progress of these operations using a responsive web interface.

## Technologies Used
- Flask - For the web framework.
- Bootstrap - For styling and responsive design.
- OpenCV - For image processing.
- MTCNN - For face detection.
- Python - For backend logic and processing.
- JavaScript - For dynamic updates on the web interface.

## Features Implemented
- **Image Scraping:**
  - Users can input a celebrity's name and the number of images to download.
  - Images are scraped from Bing and stored in a "raw_dataset" folder.
  - A progress bar shows the download progress.

- **Face Detection:**
  - Users can select a folder of a celebrity to process images and extract faces.
  - Faces are detected using MTCNN and saved in a "processed" folder.
  - The progress of face extraction is shown with two progress bars: one for the current celebrity and one for overall progress.

- **Progress Tracking:**
  - Progress bars are used to indicate the status of image downloading and face extraction.
  - Progress updates are fetched dynamically via AJAX requests.

- **Folder Management:**
  - If a folder for a celebrity already exists in the "processed" directory, a warning is shown.
  - Users have the option to continue processing or skip the existing folder.

- **User Interface:**
  - A button to return to the home page is provided after the completion of processes.
  - The interface is designed using Bootstrap for a clean and responsive look.

## Steps to Proceed
1. Fix Progress Bar Issues.
2. Correct Folder Check Logic.
3. Implement Dynamic Warning Display.

## Next Steps
- Train the model using the prepared data.
- Implement real-time face detection.

## How to Run
1. Clone the repository.
2. Install the required dependencies.
3. Run the Flask application.

