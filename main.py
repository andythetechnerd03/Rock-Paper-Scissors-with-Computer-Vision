from tkinter import *
import cv2
from data_preprocessing.hand_detect_mediapipe import handTracker
from keras.models import load_model
import numpy as np
import random
import emoji

rps = {0:'rock', 1:'paper', 2:'scissors'}
rock_emoji = emoji.emojize(":raised_fist:")
paper_emoji = emoji.emojize(":raised_hand:")
scissors_emoji = emoji.emojize(":victory_hand:")
rps_emoji = {0: rock_emoji, 1: paper_emoji, 2: scissors_emoji}


class Window(Tk):
    def __init__ (self):
        super().__init__()
        self.wm_title(f"Rock Paper Scissors")
        self.geometry("960x540")
        self.tracker = handTracker()

        # add buttons
        play_button = Button(self, text="Let's play!", command=self.main)
        play_button.grid(row=4, column=0)

        quit_button = Button(self, text="Quit game", command=self.quit_game)
        quit_button.grid(row=4, column=1)

        # add a label
        self.label = Label(self, text=f'Welcome to Rock Paper Scissors game {rock_emoji} {paper_emoji} {scissors_emoji}')
        self.label.grid(row=0,column=0)

        self.label_2 = Label(self, text="To play, click 'Let's play' and show us your hand. Press 'Enter' or 'Spacebar' to confirm your choice")
        self.label_2.grid(row=1,column=0)

        self.winner_label = Label(self, text="Result: ", width=50, height=10, anchor='center', font=(25))
        self.winner_label.grid(row=3, column=0)

        # add boxes
        self.human_box = LabelFrame(self, text='You', height='200', width='200')
        self.human_box.grid(row=2, column=0)
        self.computer_box = LabelFrame(self, text='Computer', height='200', width='200')
        self.computer_box.grid(row=2, column=1)
        self.human_box_text = Label(self.human_box, text="",font=(100))
        self.human_box_text.grid(row=0,column=0)
        self.computer_box_text = Label(self.computer_box, text="",font=(100))
        self.computer_box_text.grid(row=0,column=0)
        self.human_choice = None
        self.human_choice_int = None
        self.computer_choice = None
        self.computer_choice_int = None

        # video streaming
        self.cap = cv2.VideoCapture(0)
    
        # load the Rock-Paper-Scissors model pre-trained
        self.model = load_model('model/rps.h5')

    def main(self):
        self.human_choice_int, self.human_choice = self.show_frames()
        if self.human_choice_int is None: return
        self.computer_choice_int, self.computer_choice = self.machine_rps()
        self.human_box_text.config(text=rps_emoji[self.human_choice_int]+self.human_choice)
        self.computer_box_text.config(text=rps_emoji[self.computer_choice_int]+self.computer_choice)
        winner = self.determine_winner()
        if winner == 0: 
            self.winner_label.config(text=f"Result: You win! Congratulations {emoji.emojize(':winking_face:')}")
            self.winner_label.config(fg='green')
        elif winner == 1:
            self.winner_label.config(text=f"Result: Tough luck! Try again next time {emoji.emojize(':disappointed_face:')}")
            self.winner_label.config(fg='red')
        else:
            self.winner_label.config(text=f"Result: That is a tie! Whew {emoji.emojize(':handshake:')}")
            self.winner_label.config(fg='blue')

    def show_frames(self):
        self.cap.open(0)
        while True:
            success,frame = self.cap.read()
            image = self.tracker.handsFinder(frame)
            lmList = self.tracker.positionFinder(image)
            self.human_choice_int, self.human_choice = None, None
            # check if there are any hands at all
            if lmList:
                coords = self.encode_coords(lmList)
                # predict the class of the hand
                label = self.model.predict(coords, verbose=0)
                self.human_choice_int = np.argmax(label, axis=-1)[0]
                prob = np.round(label[:,self.human_choice_int][0])
                self.human_choice = rps[self.human_choice_int]
                # put label text on the screen
                text = f'{self.human_choice}: {prob}'
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, text ,(50, 50), font, 1, (0, 255, 255), 2, cv2.LINE_4)
            cv2.imshow("Video",image)
            # Press SPACEBAR to end the video stream & register hand
            if cv2.waitKey(1) & 0xFF == 32:
                break
        self.cap.release()
        cv2.destroyAllWindows()

        return self.human_choice_int, self.human_choice

    def encode_coords(self,lmList): # turn coords nested lists into a 42-dimensional vector to be fed into the model
        coords = np.zeros((1,63)) # initialize a vector of 63 elements - x,y,z of 21 hand landmarks
        for coord in lmList:
            idx = coord[0]
            coords[:,idx*3] = coord[1] # x
            coords[:,idx*3+1] = coord[2] # y
            coords[:,idx*3+2] = coord[3] # z
        return coords

    def machine_rps(self): # generate random choice of rock-paper-scissors of the computer
        # 0 - rock, 1 - paper, 2- scissors
        self.computer_choice_int = random.randint(0,2)
        self.computer_choice = rps[self.computer_choice_int]

        return self.computer_choice_int, self.computer_choice
    
    def determine_winner(self): # determine the winner by comparing choices (int)
        winner = None # 0 if human wins, 1 if machine wins, 2 if it's a tie
        if (self.human_choice_int == 0 and self.computer_choice_int == 2) or (self.human_choice_int == 1 and self.computer_choice_int == 0) or (self.human_choice_int == 2 and self.computer_choice_int == 1):
            winner = 0
        elif (self.human_choice_int == 2 and self.computer_choice_int == 0) or (self.human_choice_int == 0 and self.computer_choice_int == 1) or (self.human_choice_int == 1 and self.computer_choice_int == 2):
            winner = 1
        else: winner = 2
        return winner

    def quit_game(self):
        self.destroy()
        

if __name__ == "__main__":
    app = Window()
    app.mainloop()
