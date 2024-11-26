# Immersion_Short_Video

These are the codes and data for the paper:

**Zhiyu He, Shaorun Zhang, Peijie Sun, Jiayu Li, Xiaohui Xie, Min Zhang and Yiqun Liu. Understanding User Immersion in Online Short Video Interaction. 32nd ACM International Conference on Information and Knowledge Management (CIKM 2023). **

## Questionnaires

"questionnaires/qustionnaires.pdf" shows the survey, the session immersion scale, and the satisfaction scale.

"questionnaires/instruction.md" is the instruction for video-level self-assessment in the lab study. 

## Data

### labels

The data is organized in the form of dictionary

Path = "./data/metadata/{uid}_behavior_MAES.json"

For each interaction(uid,iid), fields:

"start_time", "end_time": the start and end time of viewing the video (timestamp)
"eeg_start_time", "eeg_end_time": the EEG time aligned with start and end time
"like", "like_time": like or not; if like, the timestamp of liking
"video_duration": duration of the video (in seconds)
"immersion", "satisif", "valence", "arousal": rating from 1-5
"session_id", "video_order": the order of session and video, specifically
"video_tag": encoded video tag
"session_mode": personalized / randomized / non-personalized / mixed

### EEG data

For detailed information, please visit "./data/EEG_data/". 

### partcipants

The basic information of the participants

Path = "./data/participant.csv"

Fields: iid, age, gender, usage (using year of short video app we employed, 1: never, 2:under 6 months, 3: 6-12 months, 4: 1-2 years, 5: over 2 years)

### Video features

The features of the videos

Path = "./data/video_features/video_features.csv", "./data/video_features/video_ComParE-example.csv"


## Codes

The codes of video feature extraction, EEG features extraction, and immersion prediction task.

# Reference
Zhiyu He, Shaorun Zhang, Peijie Sun, Jiayu Li, Xiaohui Xie, Min Zhang, and Yiqun Liu. 2023. Understanding User Immersion in Online Short Video Interaction. In Proceedings of the 32nd ACM International Conference on Information and Knowledge Management (CIKM '23). Association for Computing Machinery, New York, NY, USA, 731–740. https://doi.org/10.1145/3583780.3615099

If you have any problems with this dataset, please contact Zhiyu He at hezhiyu0302 AT 163.com 


