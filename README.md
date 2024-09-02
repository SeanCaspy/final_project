# final_project
Hearing better is our final project for our coputer science Bsc.
In this project we tried to create a program that simulate hearing aids that can filterout background sounds and keep only the the talking sounds around the user.

Code Description:
What the code does is to claculate the avrage volume of the sound and set it as threshold for filtering the sound since we asume a person is louder than the envirment around him. With the help of machine learing methods we cataloged the sounds, with this we added paramters to the threshold calculation to increase or decrease it.
after the calculation of the threshold we used MDCT to transformed the sound into numerical martix and compare it to the thershold. if the value was above the threshold we keep it, otherwise we multiply it by zero.
After we filltered the matrix, we transform it back to sound and stream it our to the user.
For the user iterface we used Wix website. By pressing the start butten in our website, the website sends a request for the main function to start run or stop the running.

Usage:
Some files in this project were part of the creation but we have no loger use in them. the files that are crusol for running the code are: AudioRecognition.py, Audio_processor.py, factory_noise_model.pkl, scaler.pkl.
After downloading the code run audio_processor. This will run the local server for the project. go to the website https://shirahorovitz178.wixsite.com/hearingbetter/login and log in. after login in you will see start and stop buttens. start to run the code and stop will stop it.
An eaiser way to run the code is to run "webless.py"

