Greedy Layer-wise Fine-tunning (GLa)

| **Method**                           | Fix Classifier | TrainLayer | Train Loss | **Val Loss** | ARC   |  ARC_norm    |
| :----------------------------------- | -------------- | ---------- | ---------- | ------------ | ----- |  --------    |
| Baseline                             |                | None       |            |              |       |              |
| Lora                                 | &#10004;       | All-32     | 0.81       |              | 52.99 |              |
| Lora                                 | &#10004;       | Last-16    | 0.87       |              | 52.90 |              |
| Lora                                 | &#10004;       | Last-4     | 0.94       |              | 51.62 |              |
| Lora                                 | &#10004;       | Last-1     | 1.11       |              | 49.57 |              |
| Lora                                 | &#10007;       | Last-1     | 0.75       |              | 49.74 |              |
| GLa                                  | &#10007;       | Last-4     | 0.941      | 0.951        | 46.08 |              |
| GLa with scaled lr [0.4,0.6,0.8,1.0] | &#10007;       | Last-4     | 0.943      | 0.948        | 46.50 |              |
| GLa with duplicated heads            | &#10007;       | Last-4     | 0.925      | 0.946        | 46.50 |              |
| GLa with duplicated heads            | &#10004;       | Last-4     | 1.104      | 1.040        | 46.59 |              |
| GLa with pretrained heads            | &#10007;       | Last-4     |            |              |       |              |
| GLa with pretrained heads            | &#10004;       | Last-4     | 0.932      | 0.947        | 47.01 |              |
|                                      |                |            |            |              |       |              |
|                                      |                |            |            |              |       |              |
|                                      |                |            |            |              |       |              |
|                                      |                |            |            |              |       |              |
|                                      |                |            |            |              |       |              |
|                                      |                |            |            |              |       |              |
|                                      |                |            |            |              |       |              |
|                                      |                |            |            |              |       |              |





 