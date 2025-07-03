## This file include extra interaction that was needed to make the things work

## Entry 4 - Fix QAction Import

**Summary:** Copy+paste the reported terminal error to chatgpt (o4-mini-high model). selected the fix i liked the most. proceed to generate a new task on CODEX, loged as Entry 4 in CODEXLOG.md. Bug resolved.   

## Entry 5 - Display Image at Native Size
**Summary:** I did not like that the image was shown to small, asked to show the image at the loaded image size. the main window did not readjust the size...  

## Entry 6 - Expand Window to Image Size
**Summary:** Just asked that. Worked as intended now 


## Entry 10 - CI Workflow
**Summary:** Add by hand the pytest-cov to the requirements.txt file


## Entry 11 - ML Segmentation Toggle
**Summary:** After entry 11 i manually updated the plan based on chatgpt recommendations in which i described missing features

## Entry 15 - Metrics Panel
**Summary:** Forgot to update the PLAN.md that the task was done,
## Entry 17 - Propose Enhancements
**Summary:** Did not like it, made my own list using chatgpt and 2 images

## Entry 20 - Pixel-to-mm Calibration
**Summary:** While it worked, it was not as i wanted. i generated a new task with a better explanation of what i wanted

## Entry 21 - Calibration UI Tweaks
**Summary:** The automatic detection was not working as intended. 

## Entry 23 - ROI Selection for Volume
**Summary:** Generated some indent errors that i fixed by hand

## Entry 24 - Apex and Contact Marking
**Summary:** I am not seeing this... add some steps to the plan based on what i think is missing

## Entry 25 - External Contour Only
**Summary:** I found that the volume it calculate is the region of interest... add a file to define what is a droplet

## Entry 27 - Calculate & Draw Buttons
**Summary:** There are several mistakes in the calculation, but i will review at the end of the improvements tasks

## Entry 33 - Refactor droplet property functions
**Summary:** To solve the issues, i generated a prompt using ChatGPT o3 model and the issues descriptions (uploading properties.py file) and used this prompt on CODEX to generate the task to refactor the functions

## Entry 37 - Drop mode classification and ## Entry 38 - Display drop mode in GUI
**Summary:** To solve add a feature to classify if the droplet is pendant or sessile, i generated a prompt using ChatGPT o3 model and  used this prompt on CODEX to generate the task to add this function. it was made in the same task in 2 promps (chatgpt and add to gui)

## Entry 41 - Pendant detector uses base silhouette
**Summary:** The previus pull broke some nice features, i asked CODEX to use the original detector function since that worked

## Entry 46 - GUI drop analysis integration
**Summary:** The new pipeline do not save the variables of the needle region and the drop region to keep the square drawed and to use it to detect the needle. 

## Entry 47 - Persist drop analysis regions
**Summary:** Included a more detailed description on how the needle should be detected and asked to avoid removing the ROI square

## Entry 48 - Drop overlay fixes and needle width
**Summary:** Did not like how the needle was shown