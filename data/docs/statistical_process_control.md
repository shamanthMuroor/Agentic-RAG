# Statistical process control

Source: https://en.wikipedia.org/wiki/Statistical_process_control (CC BY-SA 4.0)

Statistical process control (SPC) or statistical quality control (SQC) is the application of statistical methods to monitor and control the quality of a production process. This helps to ensure that the process operates efficiently, producing more specification-conforming products with less waste scrap. SPC can be applied to any process where the "conforming product" (product meeting specifications) output can be measured. Key tools used in SPC include run charts, control charts, a focus on continuous improvement, and the design of experiments. An example of a process where SPC is applied is manufacturing lines.
SPC must be practiced in two phases: the first phase is the initial establishment of the process, and the second phase is the regular production use of the process. In the second phase, a decision of the period to be examined must be made, depending upon the change in 5M&E conditions (Man, Machine, Material, Method, Movement, Environment) and wear rate of parts used in the manufacturing process (machine parts, jigs, and fixtures).
An advantage of SPC over other methods of quality control, such as "inspection", is that it emphasizes early detection and prevention of problems, rather than the correction of problems after they have occurred.
In addition to reducing waste, SPC can lead to a reduction in the time required to produce the product. SPC makes it less likely the finished product will need to be reworked or scrapped.


== History ==
Statistical process control was pioneered by Walter A. Shewhart at Bell Laboratories in the early 1920s. Shewhart developed the control chart in 1924 and the concept of a state of statistical control. Statistical control is equivalent to the concept of exchangeability developed by logician William Ernest Johnson also in 1924 in his book Logic, Part III: The Logical Foundations of Science. Along with a team at AT&T that included Harold Dodge and Harry Romig he worked to put sampling inspection on a rational statistical basis as well. Shewhart consulted with Colonel Leslie E. Simon in the application of control charts to munitions manufacture at the Army's Picatinny Arsenal in 1934. That successful application helped convince Army Ordnance to engage AT&T's George D. Edwards to consult on the use of statistical quality control among its divisions and contractors at the outbreak of World War II.
W. Edwards Deming invited Shewhart to speak at the Graduate School of the U.S. Department of Agriculture and served as the editor of Shewhart's book Statistical Method from the Viewpoint of Quality Control (1939), which was the result of that lecture. Deming was an important architect of the quality control short courses that trained American industry in the new techniques during WWII. The graduates of these wartime courses formed a new professional society in 1945, the American Society for Quality Control, which elected Edwards as its first president. Deming travelled to Japan during the Allied Occupation and met with the Union of Japanese Scientists and Engineers (JUSE) in an effort to introduce SPC methods to Japanese industry.


=== 'Common' and 'special' sources of variation ===

Shewhart read the new statistical theories coming out of Britain, especially the work of William Sealy Gosset, Karl Pearson, and Ronald Fisher. However, he understood that data from physical processes seldom produced a normal distribution curve (that is, a Gaussian distribution or 'bell curve'). He discovered that data from measurements of variation in manufacturing did not always behave the same way as data from measurements of natural phenomena (for example, Brownian motion of particles). Shewhart concluded that while every process displays variation, some processes display variation that is natural to the process ("common" sources of variation); these processes he described as being in (statistical) control. Other processes additionally display variation that is not present in the causal system of the process at all times ("special" sources of variation), which Shewhart described as not in control.


=== Application to non-manufacturing processes ===
Statistical process control is appropriate to support any repetitive process, and has been implemented in many settings where for example ISO 9000 quality management systems are used, including financial auditing and accounting, IT operations, health care processes, and clerical processes such as loan arrangement and administration, customer billing etc. Despite criticism of its use in design and development, it is well-placed to manage semi-automated data governance of high-volume data processing operations, for example in an enterprise data warehouse, or an enterprise data quality management system.
In the 1988 Capability Maturity Model (CMM), the Software Engineering Institute suggested that SPC could be applied to software engineering processes. The Level 4 and Level 5 practices of the Capability Maturity Model Integration (CMMI) use this concept.
SPC has become popular in healthcare management contexts. It is now recommended for use in the UK's National Health Service and used regularly.
The application of SPC to non-repetitive, knowledge-intensive processes, such as research and development or systems engineering, has encountered skepticism and remains controversial. In No Silver Bullet, Fred Brooks points out that the complexity, conformance requirements, changeability, and invisibility of software results in inherent and essential variation that cannot be removed. This implies that SPC is less effective in the software development than in, e.g., manufacturing.


== Variation in manufacturing ==
In manufacturing, quality is defined as conformance to specification. However, no two products or characteristics are ever exactly the same, because any process contains many sources of variability. In mass-manufacturing, traditionally, the quality of a finished article is ensured by post-manufacturing inspection of the product. Each article (or a sample of articles from a production lot) may be accepted or rejected according to how well it meets its design specifications, SPC uses statistical tools to observe the performance of the production process in order to detect significant variations before they result in the production of a sub-standard article.
Any source of variation at any point of time in a process will fall into one of two classes.

(1) Common causes
'Common' causes are sometimes referred to as 'non-assignable', or 'normal' sources of variation. It refers to any source of variation that consistently acts on process, of which there are typically many. This type of causes collectively produce a statistically stable and repeatable distribution over time.
(2) Special causes
'Special' causes are sometimes referred to as 'assignable' sources of variation. The term refers to any factor causing variation that affects only some of the process output. They are often intermittent and unpredictable.
Most processes have many sources of variation; most of them are minor and may be ignored. If the dominant assignable sources of variation are detected, potentially they can be identified and removed. When they are removed, the process is said to be 'stable'. When a process is stable, its variation should remain within a known set of limits. That is, at least, until another assignable source of variation occurs.
For example, a breakfast cereal packaging line may be designed to fill each cereal box with 500 grams of cereal. Some boxes will have slightly more than 500 grams, and some will have slightly less. When the package weights are measured, the data will demonstrate a distribution of net weights.
If the production process, its inputs, or its environment (for example, the machine on the line) change, the distribution of the data will change. For example, as the cams and pulleys of the machinery wear, the cereal filling machine may put more than the specified amount of cereal into each box. Although this might benefit the customer, from the manufacturer's point of view it is wasteful, and increases the cost of production. If the manufacturer finds the change and its source in a timely manner, the change can be corrected (for example, the cams and pulleys replaced).
From an SPC perspective, if the weight of each cereal box varies randomly, some higher and some lower, always within an acceptable range, then the process is considered stable. If the cams and pulleys of the machinery start to wear out, the weights of the cereal box might not be random. The degraded functionality of the cams and pulleys may lead to a non-random linear pattern of increasing cereal box weights. We call this common cause variation. If, however, all the cereal boxes suddenly weighed much more than average because of an unexpected malfunction of the cams and pulleys, this would be considered a special cause variation.


== Industry 4.0 and Artificial Intelligence ==
The advent of Industry 4.0 has broadened the scope of statistical process control from traditional manufacturing processes to modern cyber-physical and data-driven systems. The review article of Colosimo et al. (2024) note that SPC now plays a role in monitoring complex, high-dimensional, and often automated processes that characterise Industry 4.0 environments, including the use of machine learning and artificial intelligence (AI) models in production settings.
One emerging line of research applies SPC techniques to artificial neural networks and other machine learning models. Instead of directly monitoring product quality, the focus is on the detection of unreliable behavior of AI systems. For example, nonparametric multivariate control charts have been proposed to track shifts in the distribution of neural network embeddings, allowing detection of nonstationarity and concept drift without requiring labelled data. This enables real-time monitoring of deployed AI systems in industrial contexts.


== Application ==
The application of SPC involves three main phases of activity:

Understanding the process and the specification limits.
Eliminating assignable (special) sources of variation, so that the process is stable.
Monitoring the ongoing production process, assisted by the use of control charts, to detect significant changes of mean or variation.
The proper implementation of SPC has been limited, in part due to a lack of statistical expertise at many organizations.


=== Control charts ===
The data from measurements of variations at points on the process map is monitored using control charts. Control charts attempt to differentiate "assignable" ("special") sources of variation from "common" sources. "Common" sources, because they are an expected part of the process, are of much less concern to the manufacturer than "assignable" sources. Using control charts is a continuous activity, ongoing over time.


==== Stable process ====
When the process does not trigger any of the control chart "detection rules" for the control chart, it is said to be "stable". A process capability analysis may be performed on a stable process to predict the ability of the process to produce "conforming product" in the future.
A stable process can be demonstrated by a process signature that is free of variances outside of the capability index. A process signature is the plotted points compared with the capability index.


==== Excessive variations ====
When the process triggers any of the control chart "detection rules", (or alternatively, the process capability is low), other activities may be performed to identify the source of the excessive variation.
The tools used in these extra activities include: Ishikawa diagram, designed experiments, and Pareto charts. Designed experiments are a means of objectively quantifying the relative importance (strength) of sources of variation. Once the sources of (special cause) variation are identified, they can be minimized or eliminated.  Steps to eliminating a source of variation might include: development of standards, staff training, error-proofing, and changes to the process itself or its inputs.


==== Process stability metrics ====
When monitoring many processes with control charts, it is sometimes useful to calculate quantitative measures of the stability of the processes. These metrics can then be used to identify/prioritize the processes that are most in need of corrective actions. These metrics can also be viewed as supplementing the traditional process capability metrics. Several metrics have been proposed, as described in Ramirez and Runger.
They are (1) a Stability Ratio which compares the long-term variability to the short-term variability, (2) an ANOVA Test which compares the within-subgroup variation to the between-subgroup variation, and (3) an Instability Ratio which compares the number of subgroups that have one or more violations of the Western Electric rules to the total number of subgroups.


== Mathematics of control charts ==
Control charts are based on a time-ordered sequence of observations 
  
    
      
        
          X
          
            1
          
        
        ,
        
          X
          
            2
          
        
        ,
        …
        ,
        
          X
          
            t
          
        
      
    
    {\displaystyle X_{1},X_{2},\dots ,X_{t}}
  
 of a process characteristic. The monitored characteristic can be single observations, averages of samples or batches, ranges, variances, or residuals from a fitted model, depending on the application.  
A typical chart consists of:  

a center line (CL) representing the in-control mean, often estimated as

  
    
      
        
          CL
        
        =
        
          
            
              X
              ¯
            
          
        
        =
        
          
            
              1
              n
            
          
        
        
          ∑
          
            i
            =
            1
          
          
            n
          
        
        
          X
          
            i
          
        
        ,
      
    
    {\displaystyle {\text{CL}}={\bar {X}}={\tfrac {1}{n}}\sum _{i=1}^{n}X_{i},}
  
  

control limits, usually defined as

  
    
      
        
          UCL
        
        =
        
          μ
          
            0
          
        
        +
        k
        σ
        ,
        
        
          LCL
        
        =
        
          μ
          
            0
          
        
        −
        k
        σ
        ,
      
    
    {\displaystyle {\text{UCL}}=\mu _{0}+k\sigma ,\quad {\text{LCL}}=\mu _{0}-k\sigma ,}
  
  
where 
  
    
      
        
          μ
          
            0
          
        
      
    
    {\displaystyle \mu _{0}}
  
 and 
  
    
      
        σ
      
    
    {\displaystyle \sigma }
  
 denote the in-control mean and standard deviation, and 
  
    
      
        k
      
    
    {\displaystyle k}
  
 is commonly chosen as 3 (the "three-sigma rule").  
An observation 
  
    
      
        
          X
          
            t
          
        
      
    
    {\displaystyle X_{t}}
  
 falling outside the interval 
  
    
      
        [
        
          LCL
        
        ,
        
          UCL
        
        ]
      
    
    {\displaystyle [{\text{LCL}},{\text{UCL}}]}
  
 signals a potential out-of-control condition. Variants such as the cumulative sum (CUSUM) chart and the exponentially weighted moving average charts (EWMA chart) are used to improve sensitivity to small or persistent shifts.  
In many applications, however, the assumption of independent observations is violated, for example in autocorrelated time series. In such cases, the conventional control limits may produce excessive false alarms. A common solution is to fit a time series model (e.g., ARIMA) and construct a residual control chart, where the model residuals  

  
    
      
        
          
            
              
                ε
                ^
              
            
          
          
            t
          
        
        =
        
          X
          
            t
          
        
        −
        
          
            
              
                X
                ^
              
            
          
          
            t
          
        
      
    
    {\displaystyle {\hat {\varepsilon }}_{t}=X_{t}-{\hat {X}}_{t}}
  
  
are monitored instead, or to adjust the control limits accordingly. Because the residuals are designed to be approximately independent and identically distributed, standard control chart theory can be applied to them. Adjusted control limits or model-based approaches are therefore required when processes exhibit dependence.


== See also ==
ANOVA Gauge R&R – Measurement systems analysis techniquePages displaying short descriptions of redirect targets
Distribution-free control chart – Statistical process monitoring tool
Electronic design automation – Software for designing electronic systems
Industrial engineering – Branch of engineering which deals with the optimization of complex processes or systems
Process Window Index – Statistical measure that quantifies the robustness of a manufacturing processPages displaying short descriptions of redirect targets
Process capability index – Statistical measure of process capability
Quality assurance – Ways of ensuring the quality of a service or product
Reliability engineering – Sub-discipline of systems engineering that emphasizes dependability
Six sigma – Business process improvement techniquePages displaying short descriptions of redirect targets
Stochastic control – Probabilistic optimal control
Total quality management – Approach to business improvement
Isolation forest – Algorithm for anomaly detection


== References ==


== Bibliography ==


== External links ==
MIT Course - Control of Manufacturing Processes
Guthrie, William F. (2012). "NIST/SEMATECH e-Handbook of Statistical Methods". National Institute of Standards and Technology. doi:10.18434/M32189.
