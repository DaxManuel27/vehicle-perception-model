# Vehicle Perception Model
## Overview
This project implements a neural network that converts raw LiDAR range images to 3D point clouds, and detects vehicles based on those points. 
## Waymo Open Dataset:
- https://waymo.com/open/
- Used only lidar and lidar_box attributes from the perception dataset.
## Research Used For Model: 
[Lang_PointPillars_Fast_Encoders_for_Object_Detection_From_Point_Clouds_CVPR_2019_paper (1).pdf](https://github.com/user-attachments/files/24483187/Lang_PointPillars_Fast_Encoders_for_Object_Detection_From_Point_Clouds_CVPR_2019_paper.1.pdf)
## Tools
- Python
- PyTorch
- Pandas
- Numpy
- Open3d (For visualization)

## Some Examples
### Red - Ground Truth, Green - Model Predictions
<img width="653" height="469" alt="Screenshot 2026-01-08 at 1 00 37 PM" src="https://github.com/user-attachments/assets/e0278750-49ce-4d2a-a1c7-41c84ccfcea2" />
<img width="1032" height="567" alt="Screenshot 2026-01-08 at 12 59 35 PM" src="https://github.com/user-attachments/assets/57652068-9fdd-4fcb-8406-0187e167a6ff" />

