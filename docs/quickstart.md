English | [中文](https://github.com/unademo/UNet_Nested4Tiny_Objects_Keypoints/blob/master/docs/quickstart_cn.md)

## Quick Start

#### Heatmap creation

- Number of  heatmap(s) is suggested to be less than 4 (≤ 4).

- Arrange the keypoints onto different target heatmaps according to their intrinsic logical relationship.

- When validating & inferencing, the predictions, also the heatmap(s), should be 'translated' to points according to the creation.

- If not modify the "translation" method, the default "translation" method: "auto match keypoints" will be adopted, which means automatically matching the predict keypoints with target keypoints by calculating the minimum distance.

#### Run

- 

