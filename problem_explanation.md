## Improve ball tracking

We have a YOLO11x model trained with Ultralytics at weights. This model detects:
- Balls with class ID 0
- Person with class ID 1

The ball prediction is faulty as the videos that we use are not ideal. We can only change what we do in post processing.

I want to do two things:
1 - I want to delete all idle balls. Sometimes the model predicts as balls either things that are not balls or balls that are just laying on the ground. I would like this prediction to not be shown in the final product. For that I want to delete all balls predictions that haven't move much in all their prediction.

2 - I want to interpolate the ball trajectory between frames in which the ball tracking is lost.

For the people, for now I do not want to see that prediction.