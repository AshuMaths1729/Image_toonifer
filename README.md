# Image Toonifer

Generates toonified version of the given image

___

## Results
Original Image 1

![alt_text](https://github.com/AshuMaths1729/Image_toonifer/blob/master/test2.jpg "Original Image")

Toonifed Image 1
![alt_text](https://github.com/AshuMaths1729/Image_toonifer/blob/master/test2_toonified.png "Toonified Image")


Original Image 2

![alt_text](https://github.com/AshuMaths1729/Image_toonifer/blob/master/test5.jpg "Original Image")

Toonifed Image 2
![alt_text](https://github.com/AshuMaths1729/Image_toonifer/blob/master/test5_toonified.png "Toonified Image")

___
Now the Tech part.

I used:
* Applied Median Filtering.
* Calculated edges using Canny Edge Detection.
* KMeans Clustering to perform Image color quantization.
* Dilated the edges map a little bit.
* Filtered edges map using Bilateral and Median filters.
* Applied image thresholding on filtered edges map.
* Lastly draw contours on the color-quantized image.

And then hurrah, we have a somewhat tonnified version of the image.
This is just an attempt.
If anyone of you seeing this repo, has some good alterations, may pull request to help me and enlighten me.
