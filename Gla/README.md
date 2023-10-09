Greedy Layer-wise Fine-tunning (GLa)

| **Method**                           | Fix Classifier | TrainLayer | Train Loss | **Val Loss** | ARC   |
| :----------------------------------- | -------------- | ---------- | ---------- | ------------ | ----- |
| Lora                                 | &#10004;       | All-32     | 0.81       |              | 52.99 |
| Lora                                 | &#10004;       | Last-16    | 0.87       |              | 52.90 |
| Lora                                 | &#10004;       | Last-4     | 0.94       |              | 51.62 |
| Lora                                 | &#10004;       | Last-1     | 1.11       |              | 49.57 |
| Lora                                 | &#10007;       | Last-1     | 0.75       |              | 49.74 |
| GLa                                  | &#10007;       | Last-4     |            |              |       |
| GLa with scaled lr [0.4,0.6,0.8,1.0] | &#10007;       | Last-4     |            |              |       |
| GLa with duplicated heads            |                | Last-4     |            |              |       |
| GLa with duplicated heads            |                | Last-4     |            |              |       |
| GLa with pretrained heads            |                | Last-4     |            |              |       |
| GLa with pretrained heads            |                | Last-4     |            |              |       |
|                                      |                |            |            |              |       |
|                                      |                |            |            |              |       |
|                                      |                |            |            |              |       |
|                                      |                |            |            |              |       |
|                                      |                |            |            |              |       |
|                                      |                |            |            |              |       |
|                                      |                |            |            |              |       |
|                                      |                |            |            |              |       |





 