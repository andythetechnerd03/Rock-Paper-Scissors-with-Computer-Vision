# Rock Paper Scissors game using Deep Learning
This repo contains of the most classic game in history... rewritten with the help of AI and Computer Vision. You have a little fun time playing with the computer on your free time or simply want to test your luck.
## Introduction
In this repo, I have developed an executable Windows app that runs the game of Rock Paper Scissors. You will give our your choice of rock, paper, or scissors, using your hand and pointing to the webcam. It will read the hand and recognize either one of the three possibilities. Then, the computer will randomize its own choice and then the app will give you the result just like real life. </br>
- Rock beats Scissors
- Scissors beat Paper
- Paper beats Rock. </br>

Here's some screenshots of the interface:
![Screenshot 2023-07-01 095941](https://github.com/andythetechnerd03/Rock-Paper-Scissors-with-Computer-Vision/assets/101492362/8ea755f8-cf57-4c17-bb82-2d9a41a2bd48)
![Screenshot 2023-07-01 100044](https://github.com/andythetechnerd03/Rock-Paper-Scissors-with-Computer-Vision/assets/101492362/1f31a5d8-afb5-4d4e-8c13-8d12b3accf0a)
![Screenshot 2023-07-01 100056](https://github.com/andythetechnerd03/Rock-Paper-Scissors-with-Computer-Vision/assets/101492362/9156aae8-f975-493d-a480-e0439198b98a)

## Technical Details
### Interface
- The app runs on the base of Python and utilizes the Tkinter library - which is of the most popular library for developing GUI applications.
- I made use of several Tkinter components such as Labels, Frame, Button, etc.
### Model
- The model used in the app is the Hand Recognition Model build in conjuntion with Google's MediaPipe Model for Hand Landmarks detection. I chose the Hand Landmarks over the typical Convolutional Neural Network because it is much more accurate and has good generalization capabilities. Plus, it can work in various lighting conditions.
- The Hand Landmarks will output 21 hand landmarks - each with its own x,y,z coordinates. Therefore, I combined all the normalized coordinates into a tabular dataframe so that I can feed into the Neural Network. The training is done on the Kaggle platform with GPU T4 x2 for shorter inference time.
- The x,y,z coordinates are normalized - meaning they are not dependent on the position of the hand, but rather their relationship among the coordinates, which helps the model be more robust.
- The model architecture is fairly simple - a bunch of ReLU layers followed by a final Softmax layer to output the probability of each class.
- The dataset is taken on my smartphone camera and each class has roughly 1000 pictures with some augmented images using rotations or flipping.
### Deployment
- I deployed the Python file into an executable .exe file using the ```PyInstaller``` library.
## How to Install
- Make sure the latest version of Python and PIP is installed in your computer (Python 3.10 **recommended**).
- Download the entire repo (it is fairly heavy, around 2 GB) so it might take some time.
- Go to the link https://1drv.ms/f/s!AtU3D6nm_XTXzzrD1RLx5qLO42XZ?e=8p37rK to download ```/main``` folder (since it is too large upload to GitHub) and move it to the main folder
- Go into Terminal or Command Prompt with Admin, and Change directory into the directory with the repo using ```cd <your_dir>```
- Run ```pip install -r requirements.txt``` to install all required packages into the computer.
- Go into ```/main``` you just downloaded and run the ```main.exe``` file (you should run as administrator to avoid any errors).
## How to Use the App
- When launching the executable file, you will have a UI
- Click on "Let's play" to launch the webcam. (make sure you have one!)
- Use your hand to signal rock, paper or scissors in front of the camera, make sure the text appears on the screen and the landmarks appear. (this may take a while to initially boot).
- Press SPACEBAR to register the hand.
- Wait for the result, you will see your response and the computer's choice and the final result - Win, Lose or Tie.
- You can close the app using the "Quit game" button.

## Future Improvements
- The model only has 3 outputs - Rock, Paper, Scissors and not Others, so if I have my hand having a different pose than the 3 classes above, the model will misclassify as either of the three.

## Notes
- The video stream, when run for the first time, can take some time recognize the hand, so be patient.
