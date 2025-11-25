# AWS AI Practitioner certificate

## Notes from UDEMY class

## Regions and services

First off, you need an aws account, you can start one for free and have a 6-month trial and 200 usd in credits.

You have a ton of services available at AWS, but not all services are available on all regions. Make sure that the services you use exist for your region. If you need a service unavailable for your region, select a different region and be done with it.

## Responsibilities

The customer has some responsibilities and AWS has some others. Some of these responsibilities overlap.

![responsibilities](./images/responsibilities.png)

The customer is responsible for the security IN THE CLOUD.

AWS is responsible for the security OF THE CLOUD.

## Acceptance use policy

- Can't do illegal, harmful or offensive use of content
- can't do security violations
- Can't do network abuse
- Can't do any email or messages abuse

Pretty obvious rules but legal needs this stated. You can read the whole thing [here](https://aws.amazon.com/aup/).

## Udemy questions:

1. You ONLY want to manage Applications and Data. Which type of Cloud Computing model should you use?

- On-premises -> you manage everything from networking to applications.
- Infrastructure as service IaaS -> you manage the operating system, the middleware, the data and the applications.
- Software as service SaaS -> everything is managed by a 3rd party.
- `Platform as service PasS -> You only manage data and the applications.`

2. What is the pricing model of Cloud Computing?

- Discounts over time
- `Pay-as-you-go pricing -> on cloud computing you are only charged for what you use.`
- pay once a year
- Flat-rate pricing

3. Which Global Infrastructure identity is composed of one or more discrete data centers with redundant power, networking, and connectivity, and are used to deploy infrastructure?

- Edge locations -> edge locations are caching sites to deliver content to end users with lower latency. They are located in availability Zones. They are not used for deployment but for caching content.
- `availability zones -> This is the definition of availability zones`
- regions -> large geographical areas that contain multiple data centers, they are used for geographic organization and data residency purposes. They do not specify the actual physical structures like data centers that form the core of the cloud infrastructure.

4. Which of the following is NOT one of the Five Characteristics of Cloud Computing?

- Rapid elasticity and scalability
- Multi-tenancy and resource pooling
- `Dedicated support agent to help you deploy applications -> in the cloud everything is self service`
- on-demand self service

5. Which are the 3 pricing fundamentals of the AWS Cloud?

- Compute, storage, and data transfer in the AWS cloud -> transfer into the cloud is free or minimal cost.
- compute, networking, and Data transfer out of the AWS cloud -> "networking" is an umbrella that covers data transfer and other network-related configurations, which are handled slightly differently in pricing.
- `Compute, storage, and data transfer out of the AWS cloud -> these are the 3 main pillars of AWS pricing`
- Storage, functions and Data transfer in the AWS cloud. -> "functions" are a specific type of compute resource and not a broad enough category to replace "compute" entirely when discussing fundamental infrastructure components. "Compute" encompasses all processing resources, including virtual machines (EC2), containers, and serverless functions.

6. Which of the following options is NOT a point of consideration when choosing an AWS Region?

- Compliance with data governance - you may have location specific data regulations.
- Latency - you want to be as closes as possible to the end users.
- `Capacity availability - capacity is unlimited in the cloud.`
- Pricing - different regions may have varied costs for compute, storage, and data transfer.

7. Which of the following is NOT an advantage of Cloud Computing?

- train capital expense (CAPEX) for operational expense (OPEX)
- `Train your employees less - you must train your employees so they learn to use the cloud effectively`
- Go global in minutes
- Stop spending money running and maintaining data centers

8. AWS Regions are composed of?

- Two or more edge locations - edge locations are used to distribute content to users
- One or more discrete data centers - they are not physically separate i think
- Three or more availability zones - Regions are made of multiple, isolated, and physically separate Availability zones within the geographic area.

9. Which of the following services has a global scope?

- EC2 - regional service
- `IAM - global service (Identity and access management)`
- Lambda - regional service
- Rekognition - regional service

10. Which of the following is the definition of Cloud Computing?

-
