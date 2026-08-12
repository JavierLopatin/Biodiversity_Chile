# STEMP: SPATIO-TEMPORAL MODELLING PROTOCOL

# A PREPRINT

# Jan Linnenbrink

Institute of Landscape Ecology

University of Münster

48149 Münster, Germany

jan.linnenbrink@uni-muenster.de

# Jakub Nowosad

Institute of Landscape Ecology

University of Münster

48149 Münster, Germany

Institute of Geoecology and Geoinformation

Adam Mickiewicz University

61-680 Poznan, Poland´

# Marvin Ludwig

Institute of Landscape Ecology

University of Münster

48149 Münster, Germany

# Anna Frederike Jablotschkin

Institute of Landscape Ecology

University of Münster

48149 Münster, Germany

# Fabian Schumacher

Institute of Landscape Ecology

University of Münster

48149 Münster, Germany

# Teja Kattenborn

Chair of Sensor-based Geoinformatics (geosense)

University of Freiburg

79106 Freiburg, Germany

# Hanna Meyer

Institute of Landscape Ecology

University of Münster

48149 Münster, Germany

hanna.meyer@uni-muenster.de

July 24, 2026

# ABSTRACT

1. Spatio-temporal predictive machine-learning modelling is an important tool in environmental research and the geosciences. However, machine-learning models are highly sensitive to both the characteristics of the training data, such as its distribution, and methodological choices, including the cross-validation strategy and model tuning. Each decision has impact and implications on the model itself as well as the estimation of the model quality and applicability for certain purposes. Taking into account the large role of machine-learning based maps of the environment in science and their transfer into practice, transparent reporting of spatio-temporal models, ideally using standardized model protocols, is essential to enable trust, transparency and comparability. However, such protocols are currently lacking for spatio-temporal modelling.   
2. We propose STeMP (Spatio-Temporal Modelling Protocol) to fill this gap by serving two purposes: standardized reporting to understand the model functioning as well as providing guidance during the modelling process by pointing at critical decisions and parameters. The protocol is structured in three sections: Overview, Model and Prediction. The Overview section contains metadata, while the Model and Prediction sections go into detail, describing predictors, evaluation and software, and further relevant elements of the modelling workflow.   
3. The protocol definition is hosted on GitHub and accompanied by an R-package (https:// github.com/LOEK-RS/STeMP). The R-package contains a web application that can be used to fill the protocol either manually or in a semi-automated way from provided modelling objects. Warnings are returned from the protocol when common pitfalls are encountered, which may help authors as a guide through the modelling process but also support reviewers in the assessment of predictive modelling studies. Via GitHub, incorporation of contributions and feedback from the community is encouraged.

# 1 Introduction

Machine-learning methods have become increasingly popular in geoscientific spatio-temporal modelling. They allow to learn complex, non-linear and often interacting relationships without the need to make assumptions about the distribution of the data, as in statistical modelling. This leads to applications of machine-learning in diverse geoscientific fields such as climatology (Yang et al., 2024), ecology (Pichler and Hartig, 2023), remote sensing and soil sciences (Wadoux, 2025). However, machine-learning models are often perceived as "black boxes", meaning that it is hard to understand their learned relationships. Furthermore, several modelling decisions must be made that may considerably change the model, the resulting performance estimates, and its interpretation. This applies to machine-learning models in general, but spatio-temporal machine-learning models in particular. Such issues may, for example, concern the autocorrelation inherent in spatio-temporal data, which complicates data splitting for model evaluation and explanation. Failure to account for these effects can lead to substantial pitfalls like over-optimistic performance estimates, which may result in false confidence in the models and may ultimately even harm the trust in spatial predictive modelling in general (Nowosad et al., 2026).

Transparently reporting important modelling decisions, like data splitting and model selection, as well as input data characteristics is thus key so that practitioners using the model and reviewers assessing it can understand its limitations, and to evaluate if they should trust the predictions made for a given purpose (e.g., for specific regions or times or where a particular quality is required). It also helps authors of a spatio-temporal machine-learning model to explicitly evaluate their modelling decisions, and to adhere to best practices. Furthermore, while a model has usually been tied to a specific prediction in environmental modelling, there is a recent shift towards the publication of models that might be applied for a wider range of tasks (Graser et al., 2026). Transparent reporting is essential in this case to enable the user of the model to assess if a model is suitable for a particular task (e.g. specific region or time). However, such transparency is often lacking in spatio-temporal machine-learning studies in geoscientific applications. For example, information on the resolution of the predictors, the distribution of the training points, and the chosen evaluation strategy is often not transparently communicated, or they are scattered through the manuscript and the appendix (see Zurell et al., 2020). Such information, however, is essential to interpret the models’ applicability for a given task, as well as the reliability of the communicated performance statistics.

Standardized model protocols are helpful in achieving transparency and reproducibility in machine-learning studies, and thus facilitate review of models, interpreting their outcomes, and enabling their reproduction. Some model protocols exist for machine-learning models in general (e.g., Mitchell et al., 2019), and there also exist a large number of protocols in fields subject to substantial legal and institutional oversight, such as medicine (e.g., Collins et al., 2015; Mongan et al., 2020; Luo et al., 2016). In environmental modelling, there are existing definitions of model protocols for more specific applications. The ODMAP protocol (Zurell et al., 2020) was developed to describe species distribution models, while the ODE protocol (Seuru et al., 2026) is directed towards socio-ecological studies. Both build on the ODD protocol (Grimm et al., 2006), which was designed to report individual-based models in ecology. In addition to model protocols, software checklists focusing on best practices regarding the functional design, objects and naming conventions exist for (spatial) machine-learning and statistical models (e.g. the rOpenSci Statistical Software Peer Review, Ram et al., 2019). However, a standardized model protocol focusing on the field of spatio-temporal machine-learning modelling, with its unique characteristics such as spatial autocorrelation, is missing (Graser et al., 2026). This hinders transparency and misses the chance to detect and mitigate pitfalls specific to spatio-temporal machine-learning, such as inadequate data splitting or predicting outside the area where the model is applicable (Nowosad et al., 2026). Hence, such a standardized protocol tailored towards spatio-temporal machine-learning models in environmental sciences is needed to fill the gap between general machine-learning protocols and protocols specialized on very specific fields of environmental mapping.

Here we propose STeMP, a Spatio-Temporal Modelling Protocol, to fill this gap. It focuses on both, the model properties – which are important for researchers that aim to apply the model to their own task – and the documentation of the prediction, which might be more important for practitioners or scientists working with the model output (e.g. the map). It covers crucial properties of the data, which have important implications for the modelling process. STeMP is accompanied by an R-package containing a web application and R-functions to analyze protocol outputs. While the protocol and the software benefit several stakeholders involved in the modelling process, it incorporates specific features tailored to three primary groups:

• Model developers: The web application provides the functionality to spot potential issues specified by the protocol and communicate them as warnings, allowing developers to proactively address and mitigate potential errors well before peer-review or deployment.   
• Model evaluators: R-functions accompanying STeMP provide automated summaries of identified issues that enable reviewers to efficiently evaluate the most critical aspects of the model.

![](Linnenbrink et al. 2026 - STeMP - Spatio-Temporal Modelling Protocol_images/399938f451ff52f59ee12326b72347809186856539a43533df2a7b4b9f6df106.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Overview"] --> B["Model title, Author names, Contact, Study title and link, Target variable, Scientific field, Objective"]
    B --> C["Modelling"]
    C --> D["Response: Map of the training locations and area, Training domain, Acquisition, Geometry, #Samples, Range, #Classes, #Samples per class, Sampling pattern, CRS, Data sources"]
    C --> E["Predictors: Types, #Predictors, Names, Resolution, Preprocessing, Temporal alignment, Data sources"]
    C --> F["Model evaluation and selection: Model evaluation strategy, Performance metrics, Model evaluation results, Hyperparameter tuning, Predictor selection"]
    F --> G["Learning method: Model type, Algorithm, Architecture"]
    G --> H["Model interpretation: Explainability, Scientific interpretation"]
    G --> I["Model uncertainty: Biases, Sensitivity assessment, Limitations"]
    H --> J["Prediction"]
    I --> J
    J --> K["Post-Processing: Post-processing, Threshold selection"]
    L["Prediction"] --> M["Map evaluation: Map evaluation strategy, Performance metrics, Map accuracy, Uncertainty quantification"]
    M --> N["Prediction: Extent, Resolution, Map of the prediction area, Sampling pattern plot"]
    N --> O["Post-Processing: Post-processing, Threshold selection"]
```
</details>

Figure 1: The structure of STeMP. The three boxes correspond to the three sections of the protocol. The protocol is aligned to typical machine-learning workflows, which start with the response and predictor data acquisition and end with the post-processing of the predictions. Bold text shows the sub-sections of the protocol, while italic text refers to mandatory fields. See Table A1 for a complete list of all fields and corresponding short descriptions.

• Model users: Detailed documentation allows end-users to more easily assess the model’s suitability for a specific context and compare it with alternative models.

Community involvement is key to the long-term success of such a protocol. Thus, we maintain the definition of the protocol and the accompanying software on GitHub (https://github.com/LOEK-RS/STeMP), open for suggestions or discussion from the community.

# 2 Conceptual Design

Aligned with standard spatio-temporal machine-learning workflows, we structured STeMP into three main sections: Overview, Model and the optional Prediction section (Figure 1 & Table A1). The Overview section provides general information about the model (such as the author names, contact information, and if there is a publication associated with the model). The Model and Prediction sections document the decisions taken in the modelling process and the corresponding predictions. The Model section is organized in seven sub-sections that correspond to the principal stages of the modelling workflow. It covers information about the learning method – specifying the algorithm employed and whether it was used for classification or regression – together with the characteristics of the response data like the sampling design, as well as spatial and temporal resolution of the predictor variables. The section also addresses the model evaluation and selection, the interpretation of the model based on explainable AI techniques and details on potential biases or limitations of the model. Technical aspects such as the software implementation of the model can also be specified. The Prediction section is optional since models are not necessarily tied to a specific prediction or application, but may be developed to be used in different contexts and by different end-users. The prediction section covers three sub-sections: the prediction domain, the final map evaluation and uncertainty assessment, and lastly the post-processing of the resulting map. Thereby, it covers information on the method used for uncertainty quantification (e.g., quantile random forest (Meinshausen and Ridgeway, 2006) to quantify the prediction interval, or AOA (Meyer and Pebesma, 2021) and MESS (Elith et al., 2010) to delineate areas not covered by the training data), the spatio-temporal dimensions of the prediction domain, as well as post-processing measures such as threshold selection when a continuous prediction needs to be binarized.

# 3 Software accompanying STeMP

# 3.1 Functionality

STeMP is accompanied by an R-package that contains a web application and R-functions to streamline protocol completion, increase the utility of the protocol during model development, and improve its usefulness for model reviewers. The web application is designed to meet the needs of the majority of users, while the R functions offer additional functions to read and analyze completed protocols.

The web application allows many fields to be automatically filled by retrieving information from uploaded models and spatial datasets. At the moment, model objects stored as RDS objects and trained using the R-packages caret (Kuhn, 2008), tidymodels (Kuhn and Wickham, 2020) and mlr3 (Lang et al., 2019) are supported. From these objects, information about sample size, response characteristics, model parameters and others are extracted and the corresponding fields of the protocol are automatically filled. Moreover, spatial datasets can be uploaded, including the training locations and the prediction area. From these datasets, information such as the coordinate reference system can be extracted, and maps of the training locations and prediction area are automatically generated. When both training locations and prediction area are uploaded, the sampling pattern is automatically inferred based on spatial nearest-neighbour distances (see Meyer et al. 2026; Linnenbrink et al. 2024; Baddeley et al. 2015 for more details).

In addition to automation, the web application raises warnings when common spatial machine-learning pitfalls are detected (Nowosad et al., 2026). Currently, the following aspects are checked:

• Evaluation strategies not suited to the prediction task (e.g., random cross-validation with spatially clustered training points).   
• Data leakage between training and evaluation of models.   
• The absence of uncertainty quantification when the models are likely applied in extrapolation situations (i.e., when the training points are clustered).   
• The use of spatial proxies where not appropriate, or without rigorous predictor selection.

While detected issues should not automatically be interpreted as errors – as they may result from conscious and well-motivated modelling decisions – they still warrant careful consideration or justification. Providing warnings about these potential issues makes the protocol useful during model development: when the protocol is completed already at this stage, potential problems can be identified and addressed early in the modelling process.

Finally, the utility of the web application is extended by the R functions accompanying it. Most importantly, the function protocol\_analyze generates a report from a downloaded protocol. This report summarizes all persisting warnings that were raised when filling the protocol, hence providing a concise summary of potential issues encountered during the model building process. This may be particularly useful for reviewers of a model, allowing them to quickly identify and evaluate aspects requiring closer scrutiny.

# 3.2 Implementation

STeMP is implemented as a modular Shiny web application (Chang et al., 2026; Wickham, 2021), which makes the reporting workflow straightforward. We built upon the ODMAP web application (Zurell et al., 2020), which we restructured and extended to accommodate the requirements of the STeMP protocol. To ensure maintainability and scalability, we developed the web application as an R-package that follows the golem framework (Fay et al., 2026, 2021). This R-package additionally contains functions to read a downloaded protocol as a data.frame in R (protocol\_read) and to summarize the warnings that were raised by the web application (protocol\_analyze). The software is shared together with the protocol definition on GitHub (https://github.com/LOEK-RS/STeMP) under the GNU General Public License, and the version presented in this manuscript, v2026.07.00, is archived on Zenodo (https://doi.org/10.5281/zenodo.21494506). In the following paragraphs, we describe the technical design of the web application.

The code of the STeMP web application is organized into several modules, reflecting the structure of the protocol described in section 2. This modular approach not only aligns well with the conceptual design of the protocol, but also facilitates maintenance of the web application and enables re-use of code blocks. At the highest level, the run\_app function calls the two top-level modules app\_ui and app\_server. These top-level modules then invoke different modules to specify the user interface and the server logic.

The modules about and howto define the landing pages of the web application and do not contribute server logic. The remaining modules contain server logic and, at the same time, define the user interface. The modules upload and spatialdata\_upload enable uploading model objects and spatial data. The modules model\_metadata and spatialdata\_metadata automatically extract information from the uploaded objects to update the respective fields in the protocol. The protocol itself is rendered based on the module create\_protocol, which is divided into three submodules covering the three sections of the protocol: Overview, Model and Prediction. Downloading the protocol as a CSV or PDF file, or downloading the maps and plot of nearest-neighbour distances created by the web application is enabled through the sidebar module. This module also allows users to upload a previously started protocol as CSV or ZIP file by calling mod\_csv\_zip\_upload, enabling them to continue working on an existing protocol. Warning messages based on the input of the user may be raised, as defined in the warnings module. Lastly, the viewer module creates a preview of the protocol in HTML format.

![](Linnenbrink et al. 2026 - STeMP - Spatio-Temporal Modelling Protocol_images/63ce6ef4a5ae53eaee105a20617f01e0dec7887c0c970bd7375172e1f353526f.jpg)

<!-- VLM-CHART, VERIFY -->
<details>
<summary>heatmap</summary>

| Latitude | Longitude | Predicted Plant Species Richness |
| -------- | --------- | --------------------------------- |
| 10°N     | 80°W      | 50                                |
| 0°       | 70°W      | 100                               |
| 10°S     | 60°W      | 150                               |
| 20°S     | 50°W      | 200                               |
| 30°S     | 40°W      | 100                               |
| 40°S     | 30°W      | 50                                |
| 50°S     | 20°W      | 50                                |
</details>

Figure 2: The prediction resulting from the example modelling task.

We incorporated tests at three levels to cover all aspects of the web application. We utilized testthat (Wickham, 2011) for unit tests of important functions like the classification of the sampling pattern. Then, we used the testServer function from Shiny to test the reactivity inside the server functions. Lastly, we employed snapshot-based testing provided by shinytest (Chang et al., 2024) to test the general behavior of the web application, including the server functions as well as the user interface.

# 4 Example

Here we present the application of STeMP in a typical spatial predictive modelling task. In the following description, we only present the information usually included in research manuscripts: the final prediction map and the evaluation statistic. The code and data to reproduce the example, as well as a Figure showing a more complete description of the example are available as an appendix (https://doi.org/10.5281/zenodo.21493718).

We use the prediction of plant species richness in South America as an example (used to illustrate concepts in Meyer et al., 2026). What is typically written in a scientific paper to document the modelling process and the results could be the following:

We trained a random forest model based on plot-based measurements of plant species richness obtained from the sPlotOpen database (Sabatini et al., 2021) combined with spatially continuous predictors describing elevation (Jarvis et al., 2008), spatial location and the climate (WorldClim, Fick and Hijmans 2017). The model performance was high, indicated by an $R ^ { 2 }$ value of 0.7, suggesting that the resulting map is suitable to be used by planning authorities, or as an input for further modelling studies (Figure 2 shows the predicted plant species richness for South America).

However, there are several open questions that may come to mind for potential users of the map (or the model), as well as for researchers who want to understand the model or may want to compare different maps of plant species richness in South America generated from different models.

A researcher who wants to use the map as input for downstream modelling might ask:

1. How are the training points distributed in the prediction area?   
2. Is the communicated performance statistic trustworthy, and for which area does it apply?   
3. What causes the spatial patterns in the predicted map of plant species richness?

If the researcher would be interested in the plant diversity of North America, they might ask:

4. Can I use the model to make predictions for North America?

A nature conservation organization working on a more regional scale might additionally be interested in the question:

5. Can I use the map for local planning?

The aim of STeMP is to facilitate a transparent and concise reporting that ultimately enables answering these open questions. We filled the protocol using the web application. Figure 3 shows an overview over selected parts of its user interface filled with information from this example. We uploaded the trained model, as well as geospatial data of the training locations and the prediction area in the Upload data tab (see the blue rectangle in Figure 3). From these uploaded data, several fields of the protocol were automatically filled – e.g., the evaluation results and the final hyperparameter values (see the dark blue rectangle in Figure 3). Furthermore, maps of the training locations, the prediction area, as well as a plot comparing the nearest-neighbour distances encountered during prediction and between training points were automatically generated from the uploaded data. From comparing the nearest-neighbour distances, the web application automatically inferred the sampling design to be clustered. Since we used a random cross-validation in this example, a warning was then raised suggesting to assess the suitability of the chosen evaluation strategy for the clustered sampling design (see e.g. Roberts et al., 2017). After completing the protocol using the web application, the final protocol was downloaded as a PDF and CSV file (see Appendix). From this CSV file, we extracted all warnings that were raised by the web application using the function protocol\_analyze (lower part of Figure 3).

The output of the protocol and the summary of the warnings encountered while filling the protocol now enable us to answer the questions raised before:

1. The training points are clustered with regard to the prediction area (see the completed protocol in the Appendix for a map of the training points).   
2. The high $R ^ { 2 }$ value is likely overoptimistic when the model is applied to South America since the random cross-validation only tested the performance for prediction locations close to the training points.   
3. The spatial patterns are likely caused by using spatial proxies as predictors, which often lead to overfitting and linear artifacts in the prediction when used with clustered training data (Meyer et al., 2019). Since no proper predictor selection strategy was employed, the protocol shows that these issues were not mitigated (Meyer et al., 2019; Milà et al., 2024).   
4. The model was not tested for its ability to extrapolate in space, hence further testing would be needed before applying it to another region.   
5. Since the map extends over the whole of South America and the predictors have a resolution of 0.0833° (≈ 9.3 km at the equator), the map would rather be useful for continent-scale applications than for small-scale planning.

# 5 Conclusion

STeMP enables transparent reporting of important modelling decisions and data characteristics. It is tailored towards spatio-temporal machine-learning modelling, covering characteristics such as evaluation strategies, predictor selection and uncertainty assessment in light of specific characteristics such as autocorrelation inherent in spatio-temporal data. Furthermore, STeMP is implemented as a web application, which not only facilitates filling the protocol, but

# User Interface of the Web Application

STeMP v0.1

What is STeMP?

How to use this app

Upload data

Create a protocol

Protocol Viewer

![](Linnenbrink et al. 2026 - STeMP - Spatio-Temporal Modelling Protocol_images/6239f7c997424cdced4d79c78199dfa96456d3db1ca77baa73ca43d87bfe578e.jpg)

# Upload data

Upload model (.RDS)

Upload model object (.RDS)

model.RDS

Model loaded successfully

Upload geospatial data (.gpkg)

Upload sampling locations

tpoints.gpkg

Upload prediction area

modeldomain.gpkg

# Section: Overview

Model title \*

Spatial predictive modelling of plant species richness in South America

Author names \*

Anonymous

Contact \*

anonymous@xy.com

# Section: Model

Potential Issue: Random resampling might be overly optimistic with clustered training points.

Relevant sections: Model > Response; Model > Model evaluation and selection

References: Roberts et al., 2017 , Ploton et al., 2020 Milà et al., 2022

# Model evaluation and selection

Model evaluation strategy \*

Random Cross-Validation

Performance metrics \*

RMSE Rsquared

Model evaluation results \*

RMSE = 22.330, R^2 = 0.737

# Section: Prediction

Prediction domain   
Prediction map   
![](Linnenbrink et al. 2026 - STeMP - Spatio-Temporal Modelling Protocol_images/ac6486e76078fe27979b501bc2e76b45df72fd775c19bb512f223adb90303b2b.jpg)

<details>
<summary>text_image</summary>

10°N
0°
10°S
20°S
30°S
40°S
50°S
80°W 70°W 60°W 50°W 40°W
</details>

Sampling pattern plot   
![](Linnenbrink et al. 2026 - STeMP - Spatio-Temporal Modelling Protocol_images/5f1a7f6e75475f170c7ba57975ec385dd764ac3051b8737a972e9f79d87da18b.jpg)

<!-- VLM-CHART, VERIFY -->
<details>
<summary>area</summary>

| geographic distances (m) | sample-to-sample density | prediction-to-sample density |
| ------------------------ | ------------------------ | ---------------------------- |
| 1e-02                    | 0.0                      | 0.0                          |
| 1e-03                    | 0.6                      | 0.0                          |
| 1e-04                    | 0.4                      | 0.0                          |
| 1e-05                    | 0.0                      | 0.9                          |
</details>

# Create a report from the completed Protocol

library(STeMP)

protocol <- protocol\_read("protocol.csv")

protocol\_analyze(protocol)

# Warnings

Cross-Validation was used for both model selection and evaluating the final prediction   
This can lead to data leakage. See Hastie et al., 2009.   
Random Cross-Validation was used for model selection, while the training samples were clustered relative to the prediction area.   
2 This might yield overly optimistic results and thus bias model selection, or false confidence in the models' performance if no final map accuracy is estimated. See Wadoux et al., 2021, de Bruin et al., 2022, Milà et al., 2022.   
Random Cross-Validation was used for final map accuracy estimation, while the training samples were clustered relative to the   
prediction area. This might yield overly optimistic results and give false confidence in the resulting map. See Wadoux et al., 2021, de Bruin et al., 2022, Milà et al., 2022.   
4 Using spatial proxies with clustered samples likely leads to extrapolation situations. See Meyer et al., 2019, Milà et al., 2024   
The samples were clustered relative to the prediction area, which often leads to extrapolation when the model is applied to   
feature combinations not present in the training data.   
23 No methods to identify and communicate extrapolation areas were reported.   
See Meyer & Pebesma, 2021, Elith et al., 2010

Figure 3: An overview of selected parts of the web application’s user interface filled based on the example (top). The light blue rectangle shows the Upload data tab, where the trained model, the location of the training points and the prediction area were uploaded. The dark blue rectangle shows the Create a protocol tab, which is further divided into the panels Overview, Model and Prediction. Based on the uploaded data, the Prediction panel shows an automatically generated map of the prediction domain, as well as a plot of geographical nearest-neighbour distances. From the latter, a clustered sampling design is automatically inferred, leading to a warning message if a random cross-validation strategy is selected in the Model panel. The lower part of the figure shows the protocol\_analyze function, which creates a report from the completed protocol.

also renders it useful even during model development by raising warnings when common pitfalls in spatio-temporal machine-learning are encountered. Thus, STeMP fills the gap between generic model protocols and domain-specific model reports. Thereby, it covers the most important criteria as recently proposed in the vision by Graser et al. (2026).

While STeMP covers several fields and options covering temporal aspects, it is still mainly focused on spatial machinelearning. Future developments will focus on exploring ways to further extend the protocol in the temporal dimensions. Furthermore, the automatic extraction of information currently only works for model objects created in R. Future work should focus on expanding this, e.g., by supporting the more generic ONNX model format (Bai et al., 2025). Additionally, the potential of large-language models is not exploited by the current protocol. This may include extracting relevant information from existing documents (e.g., research papers), and then passing this information to STeMP. Moreover, future work may explore the possibility to automatically upload a completed protocol to Zenodo, enabling permanent storage and association with a DOI. Lastly, the web application currently enables downloading the completed protocol as CSV and PDF file or as a ZIP folder (the latter containing the figures generated during filling the protocol). However, future work may explore exporting a JSON file, which can store information more flexibly than a CSV.

We believe that the proposed spatio-temporal modelling protocol can act as a basis for a community-driven reporting standard, enabling transparency and fostering discussion about critical decisions and data characteristics in spatiotemporal modelling.

# Acknowledgements

The web application builds on the ODMAP protocol (Zurell et al., 2020), which standardizes reporting for species distribution models. We acknowledge the authors’ foundational contribution.

This project has received the financial support of the European Union’s Horizon Europe research and innovation programme under the Marie Skłodowska-Curie grant agreement No. 101147446, as well as the project BEyond within the priority Program 1374 "Biodiversity-Exploratories" (512284513) and the TRR 391 Spatio-temporal Statistics for the Transition of Energy and Transport (520388526), both funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation).

# Conflict of Interest statement

The authors declare no competing interests.

# Author Contributions

JL, HM, ML and JN conceived the ideas; JL, FS, HM, ML and JN were responsible for the conceptual design of the protocol, JL and AFJ were responsible for software implementation; JL, HM and JN led the writing of the manuscript, with TK and all other authors contributing critically to the drafts. All authors gave final approval for publication.

Table A1: The sections and fields of STeMP. Mandatory fields are shown in italic. 

<table><tr><td>Section</td><td>Field</td><td>Details</td></tr><tr><td rowspan="7">Overview</td><td>Model title</td><td>The title of the model</td></tr><tr><td>Author names</td><td>The names of the authors</td></tr><tr><td>Contact</td><td>Contact email address</td></tr><tr><td>Study title</td><td>The title of the study associated with the model</td></tr><tr><td>Study link</td><td>The link to the study associated with the model</td></tr><tr><td>Target variable</td><td>The response variable the model aims to predict</td></tr><tr><td>Scientific Field</td><td>The scientific field in which the model was developed</td></tr><tr><td rowspan="34">Model</td><td>Model type</td><td>Type of the model (e.g., classification, regression)</td></tr><tr><td>Algorithm</td><td>Name of the applied algorithm</td></tr><tr><td>Architecture</td><td>Description of the model&#x27;s architecture (nodes, layers etc.)</td></tr><tr><td>Sampling locations</td><td>Map of training locations</td></tr><tr><td>Sampling extent</td><td>The spatial/temporal extent at which the training points were collected</td></tr><tr><td>Sample acquisition</td><td>Method to collect the training points (e.g., field sampling)</td></tr><tr><td>Sample geometry</td><td>The geometry type of the training points (e.g., Polygons, Points)</td></tr><tr><td>Sample size</td><td>The number of training points</td></tr><tr><td>Classes</td><td>The number of response classes</td></tr><tr><td>Samples per class</td><td>The number of training points in each class</td></tr><tr><td>Range</td><td>The range of response values represented in the training data</td></tr><tr><td>Sampling pattern</td><td>The sampling pattern ranging from random to clustered</td></tr><tr><td>Coordinate Reference System</td><td>The coordinate reference system used</td></tr><tr><td>Data sources of response</td><td>The source(s) of the response data</td></tr><tr><td>Predictor types</td><td>Types of predictor data used to model the response</td></tr><tr><td>Number of predictors</td><td>The initial number of predictor variables</td></tr><tr><td>Names of predictors</td><td>List the names of all initial predictor variables</td></tr><tr><td>Resolution of predictors</td><td>The spatial resolution and temporal frequency/grain of the predictors</td></tr><tr><td>Preprocessing</td><td>Detail on preprocessing steps</td></tr><tr><td>Temporal alignment</td><td>Temporal alignment of the predictors with the response</td></tr><tr><td>Data sources of predictors</td><td>Data source(s) of the predictors</td></tr><tr><td>Model evaluation strategy</td><td>Details on the evaluation strategy used for model selection (e.g., random cross-validation, spatial block cross-validation, temporal block cross-validation)</td></tr><tr><td>Performance metrics</td><td>Performance metric(s) used during model evaluation</td></tr><tr><td>Model evaluation results</td><td>Results of the model evaluation</td></tr><tr><td>Hyperparameter tuning</td><td>Hyperparameter tuning methods (e.g., grid search)</td></tr><tr><td>Predictor selection</td><td>Predictor selection methods (e.g., forward-feature selection)</td></tr><tr><td>Explainability</td><td>Methods of explainable AI used to gain insights into the models behavior</td></tr><tr><td>Scientific interpretation</td><td>Discussion of the implications of the model in the context of theory</td></tr><tr><td>Potential biases</td><td>Specifications of potential biases in the data or algorithm</td></tr><tr><td>Sensitivity assessment</td><td>Assessment of the impact of varying input parameters</td></tr><tr><td>Limitations</td><td>Descriptions of limitations when applying the model</td></tr><tr><td>Software</td><td>Programming language and versioning of key software libraries</td></tr><tr><td>Code availability</td><td>Code availability (e.g., link to code repository)</td></tr><tr><td>Data availability</td><td>Data availability (e.g., link to data repository)</td></tr><tr><td rowspan="8">Prediction</td><td>Prediction extent</td><td>Spatial/Temporal extent of the prediction domain</td></tr><tr><td>Prediction resolution</td><td>The spatial resolution and temporal frequency/grain of the prediction domain</td></tr><tr><td>Map evaluation strategy</td><td>Details on the evaluation of the predicted map</td></tr><tr><td>Performance metrics</td><td>Performance metrics used to evaluate the predicted map</td></tr><tr><td>Map accuracy</td><td>Estimated map accuracy</td></tr><tr><td>Uncertainty quantification</td><td>Communication of prediction uncertainty (e.g., area of applicability)</td></tr><tr><td>Threshold selection</td><td>Details on threshold selection if the prediction was binarized</td></tr><tr><td>Post-processing</td><td>Post-processing of the prediction (e.g., spatial filters to smooth the prediction)</td></tr></table>

# References

Ruyi Yang, Jingyu Hu, Zihao Li, Jianli Mu, Tingzhao Yu, Jiangjiang Xia, Xuhong Li, Aritra Dasgupta, and Haoyi Xiong. Interpretable machine learning for weather and climate prediction: A review. Atmospheric Environment, 338:120797, 2024. ISSN 1352-2310. doi:https://doi.org/10.1016/j.atmosenv.2024.120797. URL https://www. sciencedirect.com/science/article/pii/S1352231024004722.   
Maximilian Pichler and Florian Hartig. Machine learning and deep learning—a review for ecologists. Methods in Ecology and Evolution, 14(4):994–1016, 2023. doi:https://doi.org/10.1111/2041-210X.14061. URL https: //besjournals.onlinelibrary.wiley.com/doi/abs/10.1111/2041-210X.14061.   
Alexandre M. J.-C. Wadoux. Artificial intelligence in soil science. European Journal of Soil Science, 76(2):e70080, 2025. doi:https://doi.org/10.1111/ejss.70080. URL https://bsssjournals.onlinelibrary.wiley.com/doi/ abs/10.1111/ejss.70080. e70080 EJSS-594-24.R2.   
Jakub Nowosad, Carmelo Bonannella, Darius Görgen, Marta Jemeljanova, Teja Kattenborn, Jan Linnenbrink, Hanna Meyer, Madlene Nussbaum, Luca Patelli, Rolf Simoes, and Evelyn Uuemaa. Navigating challenges in spatial machine learning: Validation, uncertainty, algorithms, and reproducibility. ERDKUNDE, Jun. 2026. doi:10.3112/erdkunde.2026.04.01. URL https://www.erdkunde.uni-bonn.de/article/view/3471.   
Anita Graser, Anahid Wachsenegger, Christos Doulkeridis, George Theodoropoulos, Bahare Salehi, Melitta Dragaschnig, Stathis Antoniou, and Yannis Theodoridis. Spatiotemporal Model Cards Enabling Future-Proof GeoAI Systems. ACM Transactions on Spatial Algorithms and Systems, 12(2):1–39, June 2026. ISSN 2374-0353, 2374-0361. doi:10.1145/3809183. URL https://dl.acm.org/doi/10.1145/3809183.   
Damaris Zurell, Janet Franklin, Christian König, Phil J. Bouchet, Carsten F. Dormann, Jane Elith, Guillermo Fandos, Xiao Feng, Gurutzeta Guillera-Arroita, Antoine Guisan, José J. Lahoz-Monfort, Pedro J. Leitão, Daniel S. Park, A. Townsend Peterson, Giovanni Rapacciuolo, Dirk R. Schmatz, Boris Schröder, Josep M. Serra-Diaz, Wilfried Thuiller, Katherine L. Yates, Niklaus E. Zimmermann, and Cory Merow. A standard protocol for reporting species distribution models. Ecography, 43(9):1261–1277, September 2020. ISSN 0906-7590, 1600-0587. doi:10.1111/ecog.04960. URL https://onlinelibrary.wiley.com/doi/10.1111/ecog.04960.   
Margaret Mitchell, Simone Wu, Andrew Zaldivar, Parker Barnes, Lucy Vasserman, Ben Hutchinson, Elena Spitzer, Inioluwa Deborah Raji, and Timnit Gebru. Model Cards for Model Reporting. In Proceedings of the Conference on Fairness, Accountability, and Transparency, pages 220–229, Atlanta GA USA, January 2019. ACM. ISBN 978-1- 4503-6125-5. doi:10.1145/3287560.3287596. URL https://dl.acm.org/doi/10.1145/3287560.3287596.   
Gary S Collins, Johannes B Reitsma, Douglas G Altman, and Karel Moons. Transparent reporting of a multivariable prediction model for individual prognosis or diagnosis (TRIPOD): the TRIPOD Statement. BMC Medicine, 13(1):1, 2015. ISSN 1741-7015. doi:10.1186/s12916-014-0241-z. URL http://www.biomedcentral.com/1741-7015/ 13/1.   
John Mongan, Linda Moy, and Charles E. Kahn. Checklist for Artificial Intelligence in Medical Imaging (CLAIM): A Guide for Authors and Reviewers. Radiology: Artificial Intelligence, 2(2):e200029, March 2020. ISSN 2638-6100. doi:10.1148/ryai.2020200029. URL http://pubs.rsna.org/doi/10.1148/ryai.2020200029.   
Wei Luo, Dinh Phung, Truyen Tran, Sunil Gupta, Santu Rana, Chandan Karmakar, Alistair Shilton, John Yearwood, Nevenka Dimitrova, Tu Bao Ho, Svetha Venkatesh, and Michael Berk. Guidelines for Developing and Reporting Machine Learning Predictive Models in Biomedical Research: A Multidisciplinary View. Journal of Medical Internet Research, 18(12):e323, December 2016. ISSN 1438-8871. doi:10.2196/jmir.5870. URL http://www.jmir.org/ 2016/12/e323/.   
Samuel Seuru, Volker Grimm, Michael Barton, Liliana Perez, Navid Mahdizadeh Gharakhanlou, Raja Sengupta, and Alejandro Miguel Dagnino. The ODE (Overview, Data, and Execution) protocol for a standardized use of machine learning in environmental, social and related interdisciplinary sciences. Environmental Modelling & Software, 198:106912, March 2026. ISSN 13648152. doi:10.1016/j.envsoft.2026.106912. URL https://linkinghub. elsevier.com/retrieve/pii/S1364815226000599.   
Volker Grimm, Uta Berger, Finn Bastiansen, Sigrunn Eliassen, Vincent Ginot, Jarl Giske, John Goss-Custard, Tamara Grand, Simone K. Heinz, Geir Huse, Andreas Huth, Jane U. Jepsen, Christian Jørgensen, Wolf M. Mooij, Birgit Müller, Guy Pe’er, Cyril Piou, Steven F. Railsback, Andrew M. Robbins, Martha M. Robbins, Eva Rossmanith, Nadja Rüger, Espen Strand, Sami Souissi, Richard A. Stillman, Rune Vabø, Ute Visser, and Donald L. DeAngelis. A standard protocol for describing individual-based and agent-based models. Ecological Modelling, 198(1-2):115–126, September 2006. ISSN 03043800. doi:10.1016/j.ecolmodel.2006.04.023. URL https://linkinghub.elsevier. com/retrieve/pii/S0304380006002043.

Karthik Ram, Carl Boettiger, Scott Chamberlain, Noam Ross, Maëlle Salmon, and Stefanie Butland. A community of practice around peer review for long-term research software sustainability. Computing in Science & Engineering, 21 (2):59–65, 2019. doi:10.1109/MCSE.2018.2882753.   
Nicolai Meinshausen and Greg Ridgeway. Quantile regression forests. Journal of machine learning research, 7(6), 2006.   
Hanna Meyer and Edzer Pebesma. Predicting into unknown space? estimating the area of applicability of spatial prediction models. Methods in Ecology and Evolution, 12(9):1620–1633, 2021. doi:https://doi.org/10.1111/2041- 210X.13650. URL https://besjournals.onlinelibrary.wiley.com/doi/abs/10.1111/2041-210X. 13650.   
Jane Elith, Michael Kearney, and Steven Phillips. The art of modelling range-shifting species. Methods in Ecology and Evolution, 1(4):330–342, 2010. doi:https://doi.org/10.1111/j.2041-210X.2010.00036.x. URL https://besjournals.onlinelibrary.wiley.com/doi/abs/10.1111/j.2041-210X.2010.00036.x.   
Max Kuhn. Building predictive models in r using the caret package. Journal of Statistical Software, 28(5):1–26, 2008. doi:10.18637/jss.v028.i05. URL https://www.jstatsoft.org/index.php/jss/article/view/v028i05.   
Max Kuhn and Hadley Wickham. Tidymodels: a collection of packages for modeling and machine learning using tidyverse principles., 2020. URL https://www.tidymodels.org.   
Michel Lang, Martin Binder, Jakob Richter, Patrick Schratz, Florian Pfisterer, Stefan Coors, Quay Au, Giuseppe Casalicchio, Lars Kotthoff, and Bernd Bischl. mlr3: A modern object-oriented machine learning framework in R. Journal of Open Source Software, dec 2019. doi:10.21105/joss.01903. URL https://joss.theoj.org/papers/ 10.21105/joss.01903.   
Hanna Meyer, Marvin Ludwig, Carles Milà, Jan Linnenbrink, and Fabian Schumacher. The CAST Package for Training and Assessment of Spatial Prediction Models, pages 247–266. Springer Nature Switzerland, Cham, 2026. ISBN 978-3- 031-99665-8. doi:10.1007/978-3-031-99665-8\_11. URL https://doi.org/10.1007/978-3-031-99665-8\_11.   
Jan Linnenbrink, Carles Milà, Marvin Ludwig, and Hanna Meyer. knndm cv: k-fold nearest-neighbour distance matching cross-validation for map accuracy estimation. Geoscientific Model Development, 17(15):5897–5912, 2024. doi:10.5194/gmd-17-5897-2024. URL https://gmd.copernicus.org/articles/17/5897/2024/.   
Adrian Baddeley, Ege Rubak, and Rolf Turner. Spatial point patterns: methodology and applications with R. CRC press, 2015.   
Winston Chang, Joe Cheng, JJ Allaire, Carson Sievert, Barret Schloerke, Garrick Aden-Buie, Yihui Xie, Jeff Allen, Jonathan McPherson, Alan Dipert, and Barbara Borges. shiny: Web Application Framework for R, 2026. URL https://shiny.posit.co/. R package version 1.13.0.9000.   
Hadley Wickham. Mastering Shiny: Build Interactive Apps, Reports, and Dashboards Powered by R. O’Reilly, 2021. ISBN 9781492047384. URL https://books.google.de/books?id=nrvAzQEACAAJ.   
Colin Fay, Vincent Guyader, Sébastien Rochette, and Cervan Girard. golem: A Framework for Robust Shiny Applications, 2026. URL https://thinkr-open.github.io/golem/. R package version 0.5.1.9015.   
Colin Fay, Sébastien Rochette, Vincent Guyader, and Cervan Girard. Engineering Production-Grade Shiny Apps. 09 2021. ISBN 9781003029878. doi:10.1201/9781003029878.   
Hadley Wickham. testthat: Get started with testing. The R Journal, 3:5–10, 2011. URL https://journal. r-project.org/articles/RJ-2011-002/.   
Winston Chang, Gábor Csárdi, and Hadley Wickham. shinytest: Test Shiny Apps, 2024. URL https://github.com/ rstudio/shinytest. R package version 1.6.0.   
Francesco Maria Sabatini, Jonathan Lenoir, Tarek Hattab, Elise Aimee Arnst, Milan Chytrý, Jürgen Dengler, Patrice De Ruffray, Stephan M. Hennekens, Ute Jandt, Florian Jansen, Borja Jiménez-Alfaro, Jens Kattge, Aurora Levesley, Valério D. Pillar, Oliver Purschke, Brody Sandel, Fahmida Sultana, Tsipe Aavik, Svetlana Aci´ c, Alicia T. R.´ Acosta, Emiliano Agrillo, Miguel Alvarez, Iva Apostolova, Mohammed A. S. Arfin Khan, Luzmila Arroyo, Fabio Attorre, Isabelle Aubin, Arindam Banerjee, Marijn Bauters, Yves Bergeron, Erwin Bergmeier, Idoia Biurrun, Anne D. Bjorkman, Gianmaria Bonari, Viktoria Bondareva, Jörg Brunet, Andraž Carni, Laura Casella, Luis Cayuela,ˇ Tomáš Cerný, Victor Chepinoga, János Csiky, Renataˇ Cušterevska, Els De Bie, André Luis de Gasper, Michele´ De Sanctis, Panayotis Dimopoulos, Jiri Dolezal, Tetiana Dziuba, Mohamed Abd El-Rouf Mousa El-Sheikh, Brian Enquist, Jörg Ewald, Farideh Fazayeli, Richard Field, Manfred Finckh, Sophie Gachet, Antonio Galán-de Mera, Emmanuel Garbolino, Hamid Gholizadeh, Melisa Giorgis, Valentin Golub, Inger Greve Alsos, John-Arvid Grytnes, Gregory Richard Guerin, Alvaro G. Gutiérrez, Sylvia Haider, Mohamed Z. Hatim, Bruno Hérault, Guillermo Hinojos Mendoza, Norbert Hölzel, Jürgen Homeier, Wannes Hubau, Adrian Indreica, John A. M. Janssen, Birgit

Jedrzejek, Anke Jentsch, Norbert Jürgens, Zygmunt K ˛acki, Jutta Kapfer, Dirk Nikolaus Karger, Ali Kavgacı, Elizabeth Kearsley, Michael Kessler, Larisa Khanina, Timothy Killeen, Andrey Korolyuk, Holger Kreft, Hjalmar S. Kühl, Anna Kuzemko, Flavia Landucci, Attila Lengyel, Frederic Lens, Débora Vanessa Lingner, Hongyan Liu, Tatiana Lysenko, Miguel D. Mahecha, Corrado Marcenò, Vasiliy Martynenko, Jesper Erenskjold Moeslund, Abel Monteagudo Mendoza, Ladislav Mucina, Jonas V. Müller, Jérôme Munzinger, Alireza Naqinezhad, Jalil Noroozi, Arkadiusz Nowak, Viktor Onyshchenko, Gerhard E. Overbeck, Meelis Pärtel, Aníbal Pauchard, Robert K. Peet, Josep Peñuelas, Aaron Pérez-Haase, Tomáš Peterka, Petr Petˇrík, Gwendolyn Peyre, Oliver L. Phillips, Vadim Prokhorov, Valerijus Rašomavicius, Rasmus Revermann, Gonzalo Rivas-Torres, John S. Rodwell, Eszter Ruprecht,ˇ Solvita Rusi¯ na, Cyrus Samimi, Marco Schmidt, Franziska Schrodt, Hanhuai Shan, Pavel Shirokikh, Jozef Šibík, Urban Šilc, Petr Sklenáˇr, Željko Škvorc, Ben Sparrow, Marta Gaia Sperandii, Zvjezdana Stanciˇ c, Jens-Christian ´ Svenning, Zhiyao Tang, Cindy Q. Tang, Ioannis Tsiripidis, Kim André Vanselow, Rodolfo Vásquez Martínez, Kiril Vassilev, Eduardo Vélez-Martin, Roberto Venanzoni, Alexander Christian Vibrans, Cyrille Violle, Risto Virtanen, Henrik von Wehrden, Viktoria Wagner, Donald A. Walker, Donald M. Waller, Hua-Feng Wang, Karsten Wesche, Timothy J. S. Whitfeld, Wolfgang Willner, Susan K. Wiser, Thomas Wohlgemuth, Sergey Yamalov, Martin Zobel, and Helge Bruelheide. splotopen – an environmentally balanced, open-access, global dataset of vegetation plots. Global Ecology and Biogeography, 30(9):1740–1764, 2021. doi:https://doi.org/10.1111/geb.13346. URL https://onlinelibrary.wiley.com/doi/abs/10.1111/geb.13346.   
Andrew Jarvis, Hannes Reuter, Andy Nelson, and Edith Guevara. Hole-filled seamless srtm data v4. International Centre for Tropical Agriculture (CIAT), 01 2008.   
Stephen E. Fick and Robert J. Hijmans. Worldclim 2: new 1-km spatial resolution climate surfaces for global land areas. International Journal of Climatology, 37(12):4302–4315, 2017. doi:https://doi.org/10.1002/joc.5086. URL https://rmets.onlinelibrary.wiley.com/doi/abs/10.1002/joc.5086.   
David R. Roberts, Volker Bahn, Simone Ciuti, Mark S. Boyce, Jane Elith, Gurutzeta Guillera-Arroita, Severin Hauenstein, Josés J. Lahoz-Monfort, Boris Schröder, Wilfried Thuiller, David I. Warton, Brendan A. Wintle, Florian Hartig, and Carsten F. Dormann. Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. Ecography, 2017. ISSN 1600-0587. doi:10.1111/ecog.02881.   
Hanna Meyer, Christoph Reudenbach, Stephan Wöllauer, and Thomas Nauss. Importance of spatial predictor variable selection in machine learning applications – moving from data reproduction to spatial prediction. Ecological Modelling, 411:108815, 2019. ISSN 0304-3800. doi:https://doi.org/10.1016/j.ecolmodel.2019.108815. URL https://www.sciencedirect.com/science/article/pii/S0304380019303230.   
Carles Milà, Marvin Ludwig, Edzer Pebesma, Cathryn Tonne, and Hanna Meyer. Random forests with spatial proxies for environmental modelling: opportunities and pitfalls. EGUsphere, 2024:1–30, 2024. doi:10.5194/egusphere-2024-138. URL https://egusphere.copernicus.org/preprints/2024/egusphere-2024-138/.   
Junjie Bai, Fang Lu, Ke Zhang, et al. Onnx: Open neural network exchange. https://github.com/onnx/onnx, 2025.