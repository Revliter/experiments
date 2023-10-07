Greedy Layer-wise Fine-tunning (GLa)

| **Method**                                                   | **Val Loss** | **Train Loss** | **ARC**   |
| ------------------------------------------------------------ | ------------ | -------------- | --------- |
| Lora (32 layers) (lora-alpaca)                               |              | 0.81           | **52.99** |
| Lora on last 16 layers (last16)                              |              | 0.87           | 52.90     |
| Lora on last 4 layers (last4)                                |              | 0.94           | 51.62     |
| Lora on last 1 layer                                         |              | 1.11           | 49.57     |
| Lora on last 1 layer + classifier                            |              | 0.75           | 49.74     |
| GLa on last 4 layers                                         |              | 1.11           | 51.28     |
| GLa on last 4 layers [0.4,0.6,0.8,1.0]                       | 0.96         | 0.90           | 48.08     |
| GLa on last 4 layers (fixed classifier with last one)        | 1.09         |                | 44.71     |
| GLa with duplicated heads (fix heads) --> may have bug?      | 1.09         |                | 44.71     |
| GLa with pretrained heads (with the original last layer cls) (fix heads) | 1.04         | 1.11           | 48.12     |
|                                                              |              |                |           |
|                                                              |              |                |           |
|                                                              |              |                |           |
|                                                              |              |                |           |
|                                                              |              |                |           |
|                                                              |              |                |           |
|                                                              |              |                |           |
|                                                              |              |                |           |
|                                                              |              |                |           |







 