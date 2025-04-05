## Project Overview

Welcome to the Convolutional Neural Networks (CNN) project!
In this project, you will learn how to build a pipeline to process real-world, user-supplied images and to put your model into an app.
Given an image, your app will predict the most likely locations where the image was taken.

By completing this lab, you demonstrate your understanding of the challenges involved in piecing together a series of models designed to perform various tasks in a data processing pipeline. 

Each model has its strengths and weaknesses, and engineering a real-world application often involves solving many problems without a perfect answer.


## Project Instructions

### Getting started

You have two choices for completing this project. You can work locally on your machine (NVIDIA GPU highly recommended), or you can work in the provided Udacity workspace that you can find in your classroom.

#### Setting up locally

This setup requires a bit of familiarity with creating a working deep learning environment. While things should work out of the box, in case of problems you might have to do operations on your system (like installing new NVIDIA drivers) that are not covered in the class. Please do this if you are at least a bit familiar with these subjects, otherwise please consider using the provided Udacity workspace that you find in the classroom.

1. Open a terminal and clone the repository, then navigate to the downloaded folder:
	
	```	
		git clone https://github.com/udacity/cd1821-CNN-project-starter.git
		cd cd1821-CNN-project-starter
	```
    
2. Create a new conda environment with python 3.7.6:

    ```
        conda create --name udacity_cnn_project -y python=3.7.6
        conda activate udacity_cnn_project
    ```
    
    NOTE: you will have to execute `conda activate udacity_cnn_project` for every new terminal session.
    
3. Install the requirements of the project:

    ```
        pip install -r requirements.txt
    ```

4. Install and open Jupyter lab:
	
	```
        pip install jupyterlab
		jupyter lab
	```

### Developing your project

Now that you have a working environment, execute the following steps:

>**Note:** Complete the following notebooks in order, do not move to the next step if you didn't complete the previous one.

1. Open the `cnn_from_scratch.ipynb` notebook and follow the instructions there
2. Open `transfer_learning.ipynb` and follow the instructions
3. Open `app.ipynb` and follow the instructions there

## Evaluation

Your project will be reviewed by a Udacity reviewer against the CNN project rubric.  Review this rubric thoroughly and self-evaluate your project before submission.  All criteria found in the rubric must meet specifications for you to pass.

## Project Submission

Your submission should consist of the github link to your repository.  Your repository should contain:
- The `landmark.ipynb` file with fully functional code, all code cells executed and displaying output, and all questions answered.
- An HTML or PDF export of the project notebook with the name `report.html` or `report.pdf`.

Please do __NOT__ include any of the project data sets provided in the `landmark_images/` folder.

### Ready to submit your project?

Click on the "Submit Project" button in the classroom and follow the instructions to submit!

## Dataset Info

The landmark images are a subset of the Google Landmarks Dataset v2.
