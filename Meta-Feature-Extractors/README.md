# Meta-Feature-Extractors
This repository contains the code for meta-feature extractors of various works in the literature. 


<h1>Meta-Features Included per Source</h1>
<details open>
<summary><h2>(1) Souto et al.</h2></summary>

- [x] Log10 No objects
- [x] Log10 Ratio of instances to features
- [x] Percentage of Missing Values
- [x] Multivariate Normality. Proportions of instances transformed with the t_squared transformation and reside withing 
      50 % of the Chi-Square distribution. 
- [x] Skewness of the t-squared vector
- [ ] ~~Type of technology used to gather data~~. This meta-feature is specific to the domain of the study it was 
      presented, therefore it is not included in this library. 
- [ ] Percentage of attributes that were kept after attribute selection filter
- [x] Percentage of outliers. Proportion of t_squared values that are more than two standard deviations distant from the 
      mean
</details>

<h2> (2) Nascimento et al.</h2>

<h2> (3) Vukicevic et al.</h2>

<details open>
<summary><h2>(4) Ferrari et al.</h2></summary>

### Attribute - Based
- [x] Log2 No objects
- [x] Log2 No attributes
- [x] Percentage of discrete attributes
- [x] Percentage of outliers
- [x] Mean entropy of discrete attributes
- [x] Mean concentration between discrete attributes
- [x] Mean absolute correlation between continuous attributes
- [x] Mean skewness of continuous attributes
- [x] Mean kurtosis of continuous attributes

</details>

