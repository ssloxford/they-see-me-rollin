# They See Me Rollin': Inherent Vulnerability of the Rolling Shutter in CMOS Image Sensors

This repository contains the evaluation source code for our paper **They See Me Rollin': Inherent Vulnerability of the Rolling Shutter in CMOS Image Sensors**.

In this paper, we present a novel attack that exploits the electronic rolling shutter as it is used in the majority of CMOS image sensors.
We show how the rolling shutter can be exploited using a bright, modulated light source (e.g., an inexpensive, off-the-shelf laser), to inject fine-grained image disruptions.
Figure 1 illustrates the row-wise acquisition of a legitimate frame and a malicious frame where light is injected during the exposure of a row, leading to distortions in a small part of the frame.

<p align="center"><img src="https://via.placeholder.com/800x200.png" width="75%"><br><em style="color: grey">Fig. 1: Illustration of the row-wise acquisition of a legitimate and a malicious frame.</em></p>

We evaluate the Rolling Shutter Attack on the use case of object detection by partially simulating it on the BDD100K and the VIRAT dataset.
We tested two well-known, state-of-the-art object detectors, namely FRCNN and SSD.
Figure 2 shows an example of how object detection is affected by the Rolling Shutter Attack. 
A network (SSD) detects objects in input images, shown with green boxes on the left. 
After overlaying the attack distortions on the image some objects are completely hidden 🟥, misplaced 🟧 or unaltered 🟩.

<p align="center"><img src="https://via.placeholder.com/800x200.png" width="75%"><br><em style="color: grey">Fig. 2: Example frames that show how object detection is affected by the Rolling Shutter Attack.</em></p>

Furthermore, we analyze the amount of distortion caused by the Rolling Shutter Attack in comparison to fully blinding, showing that for short exposure values, our attack causes interference that is similar to the expected level of perturbation seen in consecutive legitimate video frames. 
Finally, we test the effects of the Rolling Shutter Attack on autonomous driving.

## Structure of the Repository 

In order to prevent conflicts between dependencies and to facilitate the deployment, our evaluation is divided into three parts.
For each part, we provide a Docker file to create a Docker container with all the required dependencies.
In more detail, this repository is structred as follows:

#### Section 7.1 - Targeting Object Detection

The folder `object_detection` contains the source code we used in Section 7.1 - Targeting Object Detection.

#### Section 7.2 - Comparison with Blinding Attack

The folder `comparison_to_blinding` contains the source code we used in Section 7.2 - Comparison with Blinding Attack.

#### Appendix C - Effects on Autonomous Driving 

The folder `autonomous_driving` contains the source code we used in Appendix C - Effects on Autonomous Driving.

## How to use this Repository 

As mentioned above, the the entire evaluation is running in Docker containers.
Therefore, to use this repository, you will need `docker` and `docker-compose`.
The first step is to clone the entire repository:
```sh
git clone https://github.com/ssloxford/they-see-me-rollin.git
cd they-see-me-rollin/
```
Depending on the part of the evaluation you want to run, change directory:
```sh
cd <CHOOSE/EVALUATION>
```
Now just build and run the container with:
```sh
docker-compose build
docker-compose up -d
```
For more details on how to run the different evaluations, please refer to the respective README files.

## Questions, Issues or you want to contribute?

If you have any difficulties running the evaluation source code or you found a bug, please feel free to reach out to us!
We are happy to assist you.
The easiest way is to open an issue, so hopefully anyone who encounters a similar problem can find the solution.
<br>
Of course, we also welcome contributions from the community.
In that case, please fork the repository, implement your changes, test them and open a Pull Request.

## Contributors
 * [Sebastian Köhler](https://cs.ox.ac.uk/people/sebastian.kohler)
 * [Giulio Lovisotto](https://cs.ox.ac.uk/people/giulio.lovisotto)
 * [Simon Birnbach](https://cs.ox.ac.uk/people/simon.birnbach)
 * [Richard Baker](https://cs.ox.ac.uk/people/richard.baker)

## Credits

We used the following open-source projects in our evaluation:

 * [CARLA](https://github.com/carla-simulator/carla)
 * [Pylot](https://github.com/erdos-project/pylot)
