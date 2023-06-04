from hand_detect_mediapipe import handTracker
import os
import cv2
import pandas as pd

def read_n_crop_image(image_dir): #crop image to a square format then squeeze it to 256x256 standard
    image = cv2.imread(image_dir)
    width, height = image.shape[1], image.shape[0]
    crop_width = height if height<width else width
    crop_height = width if width<height else height 
    mid_x, mid_y = int(width/2), int(height/2)
    cw2, ch2 = int(crop_width/2), int(crop_height/2) 
    crop_img = image[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
    crop_img = cv2.resize(crop_img, (256,256))
    return crop_img

def get_images(dir):
    files = []
    for file in os.listdir(dir):
        if file.endswith(r".jpg"):
            files.append(file)
    return files

def detection(tracker, img):
    image = tracker.handsFinder(img)
    lmList = tracker.positionFinder(image)
    return lmList

def df_append(df, coords, label_int):
    # adding 1 row to the dataframe
    new_row = dict.fromkeys(list(range(63)) + ['label'],None)
    for coord in coords:
        idx = coord[0]
        new_row[idx*3] = coord[1]
        new_row[idx*3+1] = coord[2]
        new_row[idx*3+2] = coord[3]
    new_row['label'] = label_int
    df = df.append(new_row, ignore_index=True)
    return df

def create_df(df, train_or_test, tracker):
    j = 1
    for i,label in enumerate(['rock','paper','scissors']):
        dir = f"{train_or_test}_data/{label}/"
        images = get_images(dir)
        for image_name in images: 
            image = read_n_crop_image(dir+image_name)
            coords = detection(tracker, image)
            df = df_append(df, coords, i)
            print(f"Done image {j}")
            j+=1
    return df


def main():
    tracker = handTracker(mode=True, maxHands=1, detectionCon=0.5,modelComplexity=1)
    df_train = pd.DataFrame(columns=list(range(63)) + ['label'])
    df_train = create_df(df_train, "test", tracker)
    df_train.to_csv("test.csv")

if __name__ == "__main__":
    main()
    





