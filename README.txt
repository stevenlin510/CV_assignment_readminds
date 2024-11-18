Title: I can Read Your Mind

Description: This is an assignment from Computer Vision course in 2024 Fall semester. The ultimate goal for the program is able to determine the shape that is drawn by the user.

Main Resoures: HuggingFace.

Motivation and Finding: 
- I choose to use HuggingFace framework to deal with the problems because it provide very friendly model training environment, and also there are tons of examples can be found on the website. I think it is a good starter to use HuggingFace to train a model for the classification problem. 
- I consider the first part of this assignment's objective as video classification task, therefore I introduce the very powerful video understanding model "VideoMAE" from HuggingFace, and finetuning this model to the collected dataset.
- I have prepared 20 videos for training and 4 videos for valiation. Not suprisingly, the model achieve very bad results from our dataset. I think there is one main reason: Data is too small for this big model.
- I have tried several experiments, the model always predict the same class for all videos from different class.
- To the second part of the assignment, it is a very easy task for modern deep learning to recognize the shape, so I use zero-shot model to detect the shape from last 3 seconds of the video. (Assuming FPS is 30)
- The performance of zero-shot image classifcation is quite good.

