# Machine learning-enabled tomographic imaging of chemical short-range atomic ordering
Link: https://arxiv.org/abs/2303.13433

![image](https://user-images.githubusercontent.com/44220131/223113923-7478eb86-691f-4146-9906-b2af4f4bb67b.png)

ML-APT overview 

**Yue Li** 1*, Timoteo Colnaghi 2, Yilun Gong 1,3*, Huaide Zhang 4, Yuan Yu 4, Ye Wei 5, Bin Gan 6, Min Song 6, Andreas Marek 2, Markus Rampp 2, Siyuan Zhang 1, Zongrui Pei 7, Matthias Wuttig 4, Sheuly Ghosh 1, Fritz Körmann 1,8, Jörg Neugebauer 1, Zhangwei Wang 6*, Baptiste Gault 1,9*

1 Max-Planck-Institut für Eisenforschung GmbH, Max-Planck-Straße 1, Düsseldorf, 40237, Germany
2 Max Planck Computing and Data Facility, Gießenbachstraße 2, Garching, 85748, Germany
3 Department of Materials, University of Oxford, Parks Road, Oxford OX1 3PH, UK
4 Institute of Physics (IA), RWTH Aachen University, Aachen, 52056, Germany
5 Ecole Polytechnique Fédérale de Lausanne, School of Engineering, Rte Cantonale, 1015 Lausanne, Switzerland
6 State Key Laboratory of Powder Metallurgy, Central South University, Changsha, 410083, China
7 New York University, New York, NY10012, United States
8 Materials Informatics, BAM Federal Institute for Materials Research and Testing, Richard-Willstätter-Str. 11, 12489 Berlin, Germany
9 Department of Materials, Imperial College, South Kensington, London SW7 2AZ, UK

*Corresponding authors, yue.li@mpie.de (Y. L.); y.gong@mpie.de (Y. G.); z.wang@csu.edu.cn (Z. W.); b.gault@mpie.de (B. G.)

We developed a framework for deciphering the details of multi-type CSROs in H/MEAs, which combines ML, APT experiments/simulations, Monte-Carlo simulations, and electrical measurements. (A) First, a series of site-specific APT experiments are performed to collect the desired data, which are voxelized into millions of 1-nm cubes that are transformed into z-SDMs. (B) Then, a CSRO recognition model is obtained by utilising the simulated CSRO pattern bank to train a neural network. Its reliability is verified by a large-scale APT simulation. (C) Third, the preprocessed experimental z-SDMs are fed into the CSRO recognition model to obtain the 3D CSRO distribution. The details of multiple CSROs are revealed, supported by atomistic simulations. (D) Finally, the composition/processing-CSRO-electrical resistivity relationships are built.

The codes include three modules. 1 CSRO_pattern_bank: Generating artificial APT data along the <002> or <111> containing either a randomly distributed FCC-matrix or CSRO. 2 CNN: Training 1D convolutional neural network (CNN) to obtain an FCC-matrix/CSRO binary classification model. 3 Exp: Applications of this model in CoCrNi to obtain the 3D CSRO distributions. Here, we take Ni-Ni CSRO along <200> as an example. One experimental data under the annealing state is demonstrated. The relevant datasets are kept at https://doi.org/10.6084/m9.figshare.24274780.

System requirements:
The demo code has been present in the Google Colab platform.
