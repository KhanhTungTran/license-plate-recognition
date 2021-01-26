# License-Plate-Recognition
Basic methods:

* Determine the bounding box of the number plate with Yolo Tiny v3
* Apply segmentation algorithm to extract characters and digits from the plate
* Construct a CNN architecture to classify the characters and digits
* Postprocess the output to obtain true form results

## Quick start
```
python example.py --video_name=video_name
```

For example:

``` 
python example.py --video_name=374094.mp4
```

Folder structure:

- videos: includes some input videos

- results: consists of output videos

  The current clips do not comprise scenes of the outside street, due to the time limit of the author. If you have interest in discovering the model performance on those inputs, visit this link: https://tinyurl.com/y2v6htxf

## Dependencies

* python 3.7+
* keras
* numpy
* opencv3.x
* scikit-image
* imutils
