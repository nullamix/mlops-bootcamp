# HW03 Docker Image Size Report

| Repository           | Tag       | Size   |
|:---------------------|:----------|:-------|
| qbc12-airbnb-serving | optimized | 923MB  |
| qbc12-airbnb-serving | naive     | 1.91GB |

## Analysis
The naive image uses the full Python base image and copies the whole project into the final runtime image, so it keeps more files and layers than the service actually needs.
The optimized image uses a slim runtime stage and copies only installed dependencies plus serving source code, which makes it the better production choice because it reduces pull time, storage, and unnecessary attack surface.
