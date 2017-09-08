Filter1 : RUNTIME: 0.0328ms
Since uchar x,y,z represent BGR, swap x and y, and write z directly to output image.

Filter2 : RUNTIME: 0.8768ms
For the pixels which have 80 neighbors, simply get these 8 pixels RGB values and then assign the average value to output pixel.
Otherwise, only picked as many neigbors as possible and then average them.


Filter3 : Runtime: 0.36384ms
I used shared memory in each block, and then swap RGB values of two symmetric pixels with the help of shared memory. 

Filter4 : 
Runtime creative.jpg : 0.56ms (original picture: cat.png)
Runtime creative1.jpg: 0.574ms
		  
Render the image inside circle to black and white, the outside pixels are blurred to different extent according to their distance to black and white center. 

Or Radian blur inside, while rendering the outside pixels to be black and white.
