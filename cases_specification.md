## Case 1 

Case 1 denotes the sender continuously send available frames. A frame is regarded as "lost" if and only if, its transmission time $\geq \frac{1}{\text{FPS}}$. 

Note that under this scenario, a previous frame never affects its successors' transmission. Also, we do not send dummy data any more in all cases here. For those idle times (if any), we use concatenation.

## Case 2

Case 2 denotes that the sender sends frames, but every frames' transmission period is restricted to a specific time interval not necessarily being equally $\frac{1}{\text{FPS}}$ long. 

For some frame, if its successor used up more time, it will suffer with a shorter legit transmission interval. 

The sender always transmits the newest possible frame. 

# Case 3
Case 3, we introduced buffer (prefetch)
