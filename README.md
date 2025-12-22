# AWS AI Practitioner certificate

In this markdown file you will find notes taken from the UDEMY class "Ultimate AWS Certified AI practitioner", which you can find [here](https://www.udemy.com/share/10bvuD3@vgoH0K6J6hm6GGtFWY09gUdY3nn03pZ4S8YT3PyYdt85fnAfVMGKZbWYVfkVCJGY/). These are personal notes taken while taking the course, and are meant for reviewing prior to final test taking.

## Section 3: Introduction to AWS and cloud computing

### Regions and services

First off, you need an aws account, you can start one for free and have a 6-month trial and 200 usd in credits.

You have a ton of services available at AWS, but not all services are available on all regions. Make sure that the services you use exist for your region. If you need a service unavailable for your region, select a different region and be done with it.

### Responsibilities

The customer has some responsibilities and AWS has some others. Some of these responsibilities overlap.

![responsibilities](./images/responsibilities.png)

The customer is responsible for the security IN THE CLOUD.

AWS is responsible for the security OF THE CLOUD.

### Acceptance use policy

- Can't do illegal, harmful or offensive use of content
- can't do security violations
- Can't do network abuse
- Can't do any email or messages abuse

Pretty obvious rules but legal needs this stated. You can read the whole thing [here](https://aws.amazon.com/aup/).

### Udemy questions:

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

- Rapidly develop, test, and launch software applications - this is the definition of agility
- automatic and quick ability to acquire resources as you need them and release resources when you no longer need them - This is the definition of elasticity
- `On-demand availability of computer system resources, especially data storage (cloud storage) and computing power, without direct active management by the user`
- Change resource types when needed - this is the definition of flexibility

11. What defines the distribution of responsibilities for security in the AWS Cloud?

- AWS pricing fundamentals -> these describe the 3 fundamentals of the pay-as-you-go model
- `The shared responsibility model`
- AWS acceptance use policy - describes prohibited uses
- The AWS management console - the console allows you to manage services.

12. A company would like to benefit from the advantages of the Public Cloud but would like to keep sensitive assets in its own infrastructure. Which deployment model should the company use?

- Private cloud
- public cloud - using only the public cloud does not allow to keep sensitive assets in your own infrastructure.
- `hybrid cloud - this allows you to benefit from the flexibility, scalability, and on-demand storage access while keeping security and performance in your own infrastructure.`

13. What is NOT authorized to do on AWS according to the AWS Acceptable Use Policy?

- Building a gaming application
- Deploying a website
- `Run analytics on stolen content - you can run analytics, but not on fraudulent content.`
- Backup your data

## Section 5: Amazon bedrock and Generative AI

What is generative AI: is a subset of deep learning. It is used to generate new data that is similar to the data it was trained on... like:

- texts
- image
- audio
- code
- video
- etc etc

![genAI](./images/genai-sample.png)

The **foundation model** can do a broad range of tasks, like for example, text generation, summarization, info extraction, image genration, chatbot, QnA, etc.

We just feed a ton of data to the foundational model. They need to be trained on a wide variety of input data. It's super expensive because you need a ton of computational power and you need a ton of data.

A few big companies are making these models, like:

- OpenAI: commercial, GPT-4o
- Meta: open source llama model
- Amazon: commercial i think, Nova
- Google: BERT is open source too.
- Anthropic: commercial, claude

these companies have a lot of money and got a lot of resources.

The **LLMs** are a type of AI, relying on a foundation model, designed to generate a **coherent human-like text**.

Like ChatGPT.

They are trained on large amounts of text data. They are trained on a ton of books, articles, websites... they got billions of parameters. Super big models.

It can do language-related tasks such as translation, summarization, etc.

### How do we interact with an LLM?

Why prompts of course!!

When given a prompt the model will leverage all the existing content it has learned to generate new one.

It is non-deterministic, meaning that the same prompt will give different results. Will give similar but not exact answers.

Basically the LLM generates a list of potential words alongside probabilities, so it will try to select from that list of probable words at random.

And it will select that word, update the sentence, and do it over and over again.

#### GenAI images

cou can:

- Give a text prompt to create an image given your request
- Create an image based on another image + a prompt
- Get information from an image given a prompt

#### Diffusion models

You get a picture of a cat, and add a bit of noise again and again, until it looks just like noise. That is called a stable difussion process.

We do that for a lot of pictures, we take a bunch of images and we train the model to create noise outof these pictures.

For the generative portion, we do the opposite, and ask the model to generate an image from noise.

Then the model would just denoise and denoise until it gets a cat out the noise.

![diffusion](./images/difussion.png)

### Amazon bedrock

The AWS service we use to build genAI applications.

It's a fully managed service, you don't worry about servers. You just use the service and amazon handles the rest. You keep control of all the data you use to train the model. It's pay-per-use.

To access bedrock and all the foundation models inside it, as well as advanced features...

What type of foundation models we have access to?

- AI21 labs
- Cohere
- Anthropic
- Mistral AI
- stability ai
- amazon models

When you use any of these models, amazon bedrock will make a copy of the foundation model, and in some cases you can further finetune with your own data.

None of your data will be sent back to the providers to train their foundation model.

This is more or less how the service works, you get a playground to develop your llm agent via the foundation model, or adding rag, or actually finetuning of the model, and all is connected the the one API that you can connect to your applications.

![becrock](./images/bedrock.png)

The AWS website has a model catalog of all the models you have access to. As of november 2025 there are 252 models in the catalog.

Amazon also has a chat/text playground where you can test a particular model by trying your prompts.

The playground also gives you info like how many tokens came out, the latency, etc.

#### How to choose a model?

There is no clearcut way. Depends on needs, the model's capabilities, constraints, compliance needs, etc.

Maybe you need a multimodal model, like you need text, audio, video together as input.

You need to test

You can use **Amazon titan**, which is a high-performning foundation model from AWS. It can handle image, text, and multimodal choices via the same API from amazon bedrock.

you can also finetune the model with your own data.

Smaller models are more cost effective, but there is unsurprisingly less knowledge base.

In short it's a balancing act.

#### Quick comparison

In general Amazon titan is really cheap, but you need to make sure it works for your needs

![comparison](./images/comparisons.png)

In total right now they have about 40 providers.

Amazon **Nova Reel** is a model amazon offers for text or image to video.

On the chat/text playground, you can also try `compare mode` to see side by side two models, and help us decide on which model better suits our needs.

You check:

The cost:

- input tokens
- output tokens
- latency

And the quality of the output by having an actual look at the answer and making a judgement.

### Custom models

We can create our custom models, via the following customization methods:

- finetuning -> you provide labeled data to further finetune the model
- distillation -> you create synthetic data from a large model and use that data to finetune a smaller model.
- continued pre-training -> you use unlabeled data to pretrain a foundation model

The data basically should be stored somewhere on an s3 bucket, where you got one set of training data, and optionally one set of validation data.

You also set up the hyperparameters during finetuning (epochs, batch size, learning rate, warmup steps, etc) to ensure that the custom model is trained the way you want with the performance you want.

The output data is where you want your store your model validation outputs, and you also have to establish whether you want to give bedrock access to write to your s3 or what.

AFter you create the model, you need to pay for its usage. you pay for the `provisioned throughput`.

NOTE: Not all models can be finetuned.

#### Instruction-based fine tuning

This type of tuning uses labeled examples that are prompt-response pairs

You can use this to further train the model on a particular topic and such.

Here you can do:

- Single-turn finetuning: you give it a single turn, with the role, message, content on what you'd expect the bot to reply and get
- multi-turn finetuning: same idea, you feed it multiple turns, so for example this makes a chatbot work better.

#### continued pre-training

This is when you have unlabeled data. Also called a domain-adaptation finetuning so the model is an expert in a specific domain.

For example, you feed the entire documentation of a software so it is now the expert on that software.

It's good for industry-specific stuff like acronyms and such.

#### Notes

You should know that:

- retraining a foundation model requires moneyy.
- instruction-based is cheaper because computations are less intense and you need less data.
- you need an experienced ML engineer to do the task
- You must prepare the data, finetune, and evaluate the model
- Running a finetuned model is more expensive (that provisioned throughput bit).

#### Transfer learning

The main idea is that in this case you reuse a pretrained model for a new related task.

For example, you use claude 3, and do transfer learning to adapt it to a new task.

Very similar to finetuning, but actually it's better for image classification, and NLPs like Bert of GPT.

Actually finetuning is a specific kind of transfer learning.

#### Use Cases

You do this when you want to have a chatbot with a particular persona or tone, or geared for a specific purpose, like customer support and such.

Same if you want to use the most up-to-date information, or if you want to use exclusive data like emails, messages, etc.

Same for when you want to do a super specific task, like classification asessing accuracy.

### Evaluation

You basically need to evaluate the quality of your model via evaluation. You can build your own evaluation prompt, or use one of the default ones available on AWS, and use that to automatically get a score on performance. The way to score can be done with a bunch of different statistical methods.

After all, you want to make sure this model actually works.

#### LLM-as-judge evaluation

The setup needs some ground truth data, and the process looks like this:

![llm-evaluation](./images/llm-evaluation.png)

You need curated datasets to actually measure performance.

You may need a wide array of topics, complexity, and stuff like that.

And it's helpful to do evaluations because it gives you an idea on how accurate it is, how fast, if it scales, etc.

Some datasets allow you to evaluate bias, so you can see if your model is biased and racist.

Naturally, you can make your own benchmark datasets. But currently there are a bunch of default datasets to benchmark against, and estimate behavior scores.

#### Human evaluation

It's pretty much the same bit, of comparing the benchmark answers against the generated ones from your model. But instead of having an LLM you have a bunch of people doing that evaluation by hand.

You need subject matter experts looking at the answers to actually evaluate given their expertise.

How can humans evaluate? easy:

- thumbs up/thumbs down
- ranking
- literally just a score

The process looks like this:

![human-evaluation](./images/human-evaluation.png)

#### Common metrics

- BLEU -> you evaluate translation. Penalizes for brevity, and looks at combination of n-grams. Slightly more advanced metric.
- ROUGE -> you evaluate summarization and translation services.
  - rouge-n measures the number of matching n-grams between reference and generated text.
  - rouge-l measures the longest common subsequence between reference and generated text.
- BERTscore -> Looks at the semantic similarity between generated text. Basically you compare the meanings of the text. You use a model to compare the embeddings of a text via cosine similarity. Good at nuance.
- Perplexity -> how well the model predicts the next token. Lower is better... if it's confident it's less perplexed.

#### Business metrics

You can also evaluate the performance of a model based on business metrics like:

- user satisfaction
- average revenue per user (revenue per user attributed to genAI)
- cross-domain performance (how good the model was at multitasking)
- conversion rates (generated expected outcomes like purchases)
- efficiency (computation, resources, etc)

#### evaluation methods

- automatic
  - programmatic: we select a model and select the kind of task you want to evaluate. Metrics you can evaluate include: toxicity, accuracy, robustness, etc. you can use your own datasets or use some benchmark datasets.
  - llm-as-judge: you need to choose a model that will perform the evaluation. We have less models available. We choose if we want to evaluate a) a bedrock model, or b) a jsonl file with your 'input prompt' and your 'inference responses'.
- human
  - aws managed work team: you schedule a consultation with someone from aws
  - bring your own workforce: like the programmatic but with humans, you select the metrics to evaluate, and you can select up to 2 models to evaluate.

All these metrics connect to s3 for in/out data.

### RAG

RAG - retrieval augmented generation

We can use a foundation model that references data outside of its training data. No need for retraining the foundation model.

Basically we run a search, and the output goes into an augmented prompt. Like so:

![rag](./images/rag.png)

And so in general we chunk the data, we convert it to vectors (and there are a ton of embedding models for that) and then we store the vectors somewhere for later search and retrieval. There are a ton of options, but AWS offers the following storage options:

![dbs](./images/databases.png)

Turns out you can "chat with your document" on AWS by uploading some knowledge base, selecting the model to try, and then directly on bedrock selecting your model parameters and testing the system prompt for this rag... so you can see the quality of your result given a query.

from scratch... youd need an iam user to create a knowledge base.

You basically make an s3 bucket for your knowledge base, and then connect that bucket to your bedrock knowledge base.

If you want to stay free... use pinecone for vector storage.

But for aws ... there is the amazon opensearch serverless... they say its cost effective, but like you need a min of 2 OCU (opensearch compute unit) to run, but it takes money (.25 usd per ocu per hour... so half a dollar an hour... and use that a lot... nope.)

### genAI concepts

- tokenization - converting words to indexes or tokens
- context window - window the llm considers when answering
- embeddings - convert toxens to a high dimensionality vector, captures sentiment and context and similarities

a fun way to conceptualize embedding is to look at colors... an embedding is a bunch of colors, and two vectors with a similar hue then are similar... so for example a "dog" and a puppy would be very similar.

![embeddings](./images/embeddings.png)

Note that when people visualize these vectors they have to do dimensionality reduction because embeddings have n dimensions, but we can reliably conceptualize up to like 3 dimensions.

### Guardrails

You control the interaction between the user and the model. You filter undesirable content, remove PII, and reduce hallucinations.
You can make multiple guardrails and can monitor/analyze them to see whats up.

Amazon AWS literally has "guardrails".

you give:

- details : name, what message to show for blocked prompts, description
- configure content filters : filter harmful categories, prompt attacks
- add denied topic : name and define a topic to deny. can do several
- add word filters: profanity, custom words/phrases
- add sensitive info filter: mask any specific pii like emails and names and such, or set a certain regex pattern to filter out.
- add contextual grounding : grounding (validate that the response is based on reference source, given a grounding threshold) and relevance (validate that the model response is relevant to the user query.)

Once you select all the bits on your guardrail, you select a model and test with a prompt... and see what the model response looks like.

You will se a "model response" which is an intermediate step, which then goes after masking and then is set as a "final response". On the final response you get pii masked, and the "sorry cant do that" answer if you touch a topic you were not supposed to.

Guardrails dont seem to cost you any money. Nice.

### Agents

Agents can manage and carry out multi-step tasks related to infrastructure provisioning, deployment, and operational activities.

On AWS bedrock the agent can be an api or a lambda function + a knowledge base in s3 that saves everything in some db.

you have to give instructions to the agent on what htey can do and what they are supposed to help with.

They work like this:
![agents](./images/agents.png)

### Cloudwatch

Cloudwatch is for monitoring, they can:

- model invocation logging: sends logs of all model invocations to cloudwatch + s3 . You can analyze further and build alerts with `cloudwatch log insights`.
- cloudwatch metrics: Can use published bedrock metrics like the contentFilteredCount to see if guardrails are working. You can also build `cloudwatch alarms` on top of metrics.

#### model invocation logging

You set this up by going to `settings -> model invocation logging`.

This collects metadata of all metadata, requests, and responses for all model invocations in your account. does not apply to knowledge bases.

You can select the data types to include with the logs (text, image, embedding) and the location to save these logs (cloudwatch, s3, or both). You usually use s3 if it is too big of a log (100kb or more).

If this does not automatically create a log group, you can go to cloudwatch logs --> log groups--> create log group --> and add the name of your log group for this model logging.

So if you do the chat playground... you will see the log of that tryout. It will be some json format with the output and a ton of metadata, from model type, to tokens and latency and whatnot.

#### cloudwatch metrics

You go to `cloudwatch --> metrics --> all metrics -->bedrock`

there you can see metrics like:

- invocations
- invocation latency
- output token count

and it comes with a dashboard

### pricing

Usually you can do either on-demand or batch for cost savings. And provisioned throughput to reserve resources for a good user experience.

![pricing](./images/pricing.png)

In general the cost of projects is, from cheapest to costliest:

1. prompt engineering
2. RAG
3. Instruction-based finetuning (labeled data)
4. domain adaptation finetuning (unlabeled data)

The best way to save on cost is to use a cheap model + the least amount of tokens possible.

### Udemy questions

1. What is fine-tuning used for?

- Train a model from scratch
- `train a pre-trained foundational model on new specific data -> it improves performance in a particular task or domain`
- adapt the model's performance by tuning hyperparameters
- clean the training data to improve the model's quality

2. What type of generative AI can recognize and interpret various forms of input data, such as text, images, and audio?

- large language model
- diffusion model
- `multimodal model -> it allows a comprehensive understanding of diverse information types`
- foundation model

3. Which features can help you ensure that your model will not output harmful content?

- `Guardrails -> its a direct protective measuse`
- RAG and knowledge bases
- model evaluation -> does not implement protective strategies directly
- fine-tuning

4. You need to be able to provide always-updated data to your foundation model without retraining it. Which capability best fits your use case?

- in-context learning --> learns from examples during interaction without permanent updates. Does not use 'always updated' data
- `RAG -> it accesses real-time updated data from external databases`
- pre-training -> Does not use 'always updated' data
- fine- tuning -> does not allow continuous integration of updated data.

5. You are developing a model and want to ensure the outputs are adapted to your users. Which method do you recommend?

- automated testing --> does not resonate user needs
- using benchmark datasets --> do not reflect user context or feedback
- code review -> this is purely technical
- `human evaluation --> reflect user context or feedback`

6. Which AWS service can help store embeddings within vector databases?

- amazon s3 -> handles unstructured data but not meant for vector search
- amazon dynamo db -> handles structured data but not meant for vector search
- `amazon openseacrh serverless --> specific for vector embeddings`
- amazon kinesis data streams -> meant for real time data streaming and processing, not vector search

7. Which statement about Amazon Bedrock is INCORRECT?

- customers want access to multiple models to choose the best fit for their needs
- customers want the models fine-tuned with their data to be private
- customers do not want to manage their infrastructure - people don't want to do the whole infrastrucutre, they want to focus on models
- `customers want to build and train foundation models from scratch -> they almost always use pre-trained models`

8. What is a common use case for Amazon Bedrock?

- demand forecasting - this is just data analysis + ML
- `conversational chatbots with relevant data --> NLP and agents and rag and whatnot`
- extraction of text from images - this is just ml... bedrock is for using this as a part of a whole
- predictive maintenance of vehicles - bedrock is more about applications about dialogue and language than predictive analytics.

## Section 6: Prompt engineering

A super generic prompt gives little guidance and leaves a lot to the model’s interpretation

So to get a good response, we do 'prompt engineering', where you basically edit the prompt to enhance the outputs that fit your needs.

Two main techniques for prompt engineering:

- prompt enhancement: add things like instructions, context, examples, input data, output style, etc.
- negative prompting: give instructions on what NOT to do

See below an example of an enhanced prompt + negative prompting in red:
![prompting](./images/prompting.png)

### performance optimization

- system prompt (how the model should behave and reply)
- temperature (overall creativity//randomness of the llm): 0 to 1. higher == more creative
- top-p (percentile of words to consider based on their probabilities): 0 to 1. higher == adding more words that were not cutting it out probability wise (unlikelier words)
- top-k (how many words to consider): 1 to k. higher == more words to choose from as well.
- length (max length of the answer)
- stop requences (token sequences that signal the output should be stopped)

`Prompt latency is how fast a model responds.`

Note that prompt latency is influenced by:

- model size
- model type
- input tokens
- output tokens

And it is NOT affected by Top P, Top K, nor temperature.

### prompt engineering techniques

- zero-shot prompting : you dont give examples or explicit training, you fully rely on the model’s general knowledge. Bigger models are better.
- few-shots prompting : you provide examples to guide the output (a “few shots”) to the model to perform the task. If you provide one example only, this is also called “one-shot” or “single-shot”.
- Chain of thought (CoT) : you divide the task into a sequence of reasoning steps. On the prompt you can add something like “Think step by step”. It's particuarly helpful when this is a problem a human would actually have to divide by steps. Can be combined with Zero-Shot or Few-Shots Prompting.
- RAG : you enhance the prompt with external information that is up to date. RAG itself is not really a prompt engineering technique, BUT it's often compared to one.

## Prompt templates

You basically standardize the process of generating Prompts. You have a main prompt with a placeholder for specific inputs.

It helps with;

- processing input text and output prompts from other FMs
- orchestrates between different tools
- formats and returns the response to the user as intended.
- and of course it can be used with bedrock agents

Here's an example:

![promptTemplate](./images/promptTemplate.png)

#### Prompt injections

The downside is that if unchecked, someone can literally prompt inject a harmful command of the sort of 'ignore all previous and do this nefarious task i am asking'

To counteract that, an initial option is to literally add explicit instructions to ignore any unrelated or potential malicious content.

Not on the notes but to be honest, it's probably easiest to limit the options of user input to just a limited list, or somethign of the sort.

### Udemy questions

1. Which option accurately defines the concept of negative prompts in prompt engineering?

- Additional instructions to produce unwanted outputs
- `Examples of incorrect or undesirable outputs that the model should avoid generating --> we highlight specific outcomes that should be avoided`
- Context that should be disregarded when generating responses
- To change the overall tone in response

2. What is the temperature parameter used for?

- `Increasing the temperature makes the responses less predictable and more creative --> it increases creativity//randomness`
- Increasing the temperature makes the response more deterministic and less diverse
- Increasing the temperature will increse the final length of the answer
- Increasing the temperature will make the answers more concise

3. You are planning a complex project that involves several logical steps. You want an LLM to help you. What type of prompting should you use?

- Few-shot prompting
- Zero-shot prompting
- `Chain-of-thought prompting --> literally does step by step`

4. A task has been performed successfully in the past, and you would like your LLM to perform this task on a different input but with the same type of reasoning and output. Which type of prompting should you use?

- Zero-shot prompting
- `Few-shot prompting --> you already have a good example to give as reference, hence few-shot`
- Chain-of-thought prompting

## Section 7: Amazon Q

Amazon Q is a fully managed Gen-AI assistant for your employees trained on the entire company data.

It's built on bedrock, using multiple models, you don't really get to use the models.

It can also create and move data in the company, like create jira tickets and whatnot.

Users get authenticated via an AIM identity center. So they have only access to the documents theyre supposed to have access to.

That IAM identity center can connect to things like Google login, microsoft active directory, etc. So you can use whatever your company already uses for users.

Admin controls exist on amazon Q business, which are essentially guardrails. So you can only use internal data, block topics, and set rules both at the global and local (topic) level.

### Amazon Q business

When doing the walkthrough... user setup was easiest done via "anonymous user" but it costs 200 usd a month. And the other version of amazon Q lite needs you to create a user and tahts complicated. So this is mostly for businesses.

The way it works is:

1. you create an application (and you inmediately have a "preview" link with a frontend)
2. On data sources you add an index AND connect/add a datasource
   - index is just like a dictionary where your documents will be "indexed" and the cheap one scales up to 100k docs.
   - You add a datasource. You have multiple options, like amazon s3, github, google drive, dropbox, etc.
   - you can select a cadence for updating the data source. Or make it manual and just get a button that says "sync now"

The result from a query will come back with a 'events' which is all the steps the llm made to get you that answer, and 'sources' which is the reference docs.

You can change guardrails to allow the amazon Q to default to the llm general knowledge. But you need the fancier account.

DO NOT FORGET THERE IS AN ONGOING COST TO HAVING AN INDEX. ALWAYS DELETE THE INDEX AFTER USE.

### Amazon Q apps

This service creates Gen AI-powered apps without coding by using natural language, and still ahs access to company's internal data and can connect to plugins and whatnot.

So kinda like figma make, you just give a prompt on the type of app you want without using developers.

### Amazon Q developer

This one seems pretty neat and is a helpful tool for developers. Basically you can:

- Answer questions about the AWS documentation and AWS service selection
- Answer questions about resources in your AWS account
- Suggest CLI (Command Line Interface) to run to make changes to your account
- Helps you do bill analysis, resolve errors, troubleshooting

So for example you want to change a timeout or something, amazon q developer will give you the command you need to change the timeout.

you can also ask things like hey what was the highest cost for q1, and other neat summary stuff.

They seem to also offer an AI code companion similar to github copilot, and it supports all of the mainstream coding languages.

It gives you code suggestions and does some sort of security scan.

This thing is integrated into multiple IDEs, like VS code, visual studio, and jetbrains.

So tl;dr it does:

- Answer questions about AWS developmet
- Code completions and code generation
- Scan your code for security vulnerabilities
- Debugging, optimizations, improvements

This one costs money...

Free tier : free
pro tier: 19 usd a month, higher limits, more support

### Amazon Q bundles

They offer the following:

- Amazon Q business lite
- Amazon Q Business pro
- Amazon Q Developer pro

There is a chatbot window that you can access through the main page.

You have to agree with it having access to stuff in your account. So like if you want to list all your s3 buckets, it can go look and list them. Seems that you have a max limit of 1000 characters per query.

Fun lil fact, turns out that CLI commands you can just run in "CloudShell" which is a little button on the top right that kinda looks like ">\_"

Seems like sometimes the ai chatbot won't give commands or things that are related to "deleting" or anything that is related to security issues.

### Amazon Q for AWS services

They are adding amazon Q on top of other services

- Amazon quicksight : you visualize AWS data on dashboards, you can use Amazon Q to ask questions about your data, and create dashboards just by asking what you want.
- EC2 : So you got your virtual servers on AWS. Amazon Q gives guidance and suggestions for the right EC2 instance for your needs. you can keep talking to it to make sure your instance is a good fit as your needs evolve.
- AWS chatbot : Its a way for you to deploy an AWS chatbot in slack or microsoft teams or similar. This bot knows about your AWS account. you can access Amazon Q through this chatbot to better understand AWS services, troubleshoot issues, and identify how to fix stuff.
- AWS glue : Glue is an ETL service to extract and transfer data across places. Amazon Q can be used to answer basic questions about glue, get links to the documentation, answer questions about ETL scripts, create scripts from scratch, and troubleshoot errors.

### Partyrock

This is some playground to create genAI apps, powered by bedrock.

You don't even need an AWS account or setup. It's kinda like Amazon Q Apps.

The website has already some prebuilt apps as example.

It's basically just a neat UI where you set up your inputs, your system prompt, your model to use, and thats kinda it.

you can use just a prompt to generate an app from scratch. like figma make. So it figures out the widgets you need, how to connect them etc.

Good playground to create apps... and sells amazon bedrock and amazon q.

### Udemy questions

1. What is the main benefit of using Amazon Q Business?

- you can select the best model for your use case -> cant
- `you can easily integrate with your enterprise systems and data ->true, and apparently that is the main point`
- you have a wide range of public sources to integrate with ->true, but not main point, andnot all of them are suitable or compatible
- your ai applications can easily be exposed to your customers -> technically can do that but this is not the default, this is the option you can set up

2. What is NOT an Amazon Q Business capability?

- built-in data integrations with enterprise data sources
- plugins for enterprise applications
- `enterprise storage --> seems that you bring this, (or are using s3) not the other way around`
- Fully managed RAG capability --> offered by amazon Q

3. What is Amazon Q Developer?

- A voice-controlled assistant to perform AWS deployments
- An external team of AWS developers
- `An AI coding assistant --> kinda like copilot`

## Section 8: Artificial intelligence AI & machine learning ML

This section basically revisits basic AI/ML concepts that may come up in the exam, and delves a bit into the definitions of each.

Some notes to remember:

What is AI?

AI is is a broad field for the development of intelligent systems capable of performing tasks that typically require human intelligence. It's a broad term.

It usually has data layer --> ML framework or algorithm layer --> Model layer -->Application layer

What is ML?

ML is a type of AI for building methods that allow machines to learn. Data is leveraged to improve computer performance and make predictions. No explicit rules.

What is DL? (Deep learning)

DL Uses neurons and synapses (like our brain) to train a model. It processes more complex patterns in the data than traditional ML. It has many hidden layers. You need a lot of input data, and a lot of processing power (hence GPUs). Commonly used for NLP models and computer vision models.

What is GenAI?

It's a subset of deep learning. It's multi-purpose foundation models backed by neural networks, and can be finetuned to fit specific needs.

What is the transformer model (LLM) ?

It's a model capable to process a sentence as a whole instead of word by word. So it's faster. It gives relative importance (attention) to some parts of the sentence than others. Transformer-based LLMs include google BERT and chatGPT (Generative pretrained transformer)

TL;DR, all these bits are kinda like a human, see below:

![AIML](./images/AIML.png)

### Terms we may encounter in the exam

- **GPT (Generative Pre-trained Transformer)** – generate human text or computer code based on input prompts
- **BERT (Bidirectional Encoder Representations from Transformers)** – similar intent to GPT, but reads the text in two directions (good for translations)
- **RNN (Recurrent Neural Network)** – meant for sequential data such as time-series or text, useful in speech recognition, time-series prediction
- **ResNet (Residual Network)** – Deep Convolutional Neural Network (CNN) used for image recognition tasks, object detection, facial recognition
- **SVM (Support Vector Machine)** – ML algorithm for classification and regression (this one is the one where you have a plane that segments data into 2 classifications)
- **WaveNet** – model to generate raw audio waveform, used in Speech Synthesis
- **GAN (Generative Adversarial Network)** – models used to generate synthetic data such as images, videos or sounds that resemble the training data. Helpful for data augmentation. This is the one you give a second pass to get more training on under-exposed data.
- **XGBoost (Extreme Gradient Boosting)** – an implementation of gradient boosting

### Training data

data quality matters A LOT. garbage in == garbage out.

You got two main types:

- labeled data - you got labels to each dataset, and you can use in supervised data
- unlabeled data - you got no labels, so you're trying to find patterns or structures in your data.

types of structured data:

- tabular data - just a table rows
- time-series data - listed in a particular order

types of unstructured data:
this one has no structure to it, heavy or multimedia. Examples:

- text data - reviews or articles or something
- image data - like the ones used for object recognition tasks

### Supervised learning

There's nothing to it except that it's always using labeled data, be it continous or discrete labels. You essentially do predictions in two ways:

![supervised-learning](./images/supervised-learning.png)

- Regression: Used to predict a numeric value based on input data. The output variable is continuous, meaning it can take any value within a range
- Classification: Used to predict the categorical label of input data. The output variable is discrete, which means it falls into a specific category or class. A good method here is the k-nearest neighbors model.

#### Training vs validation vs test

Usually you bin it like this:
![train-val-test](./images/train-val-test.png)

- training : you literally use this to train your model
- validation : you use this to tune parameters and validate performance
- test : you use this to test // evaluate your performance

#### Feature engineering

Basically it's the art of making sure your features//parameters are good for ML and a good representation of behavior.

You use domain knowledge to select and transform raw data into meaningful features.

There's a couple of techniques:

- Feature Extraction – extracting useful information from raw data, such as deriving age from date of birth
- Feature Selection – selecting a subset of relevant features, like choosing important predictors in a regression model
- Feature Transformation – transforming data for better model performance, such as normalizing numerical data

How does it look like on STRUCTURED data:

For example you have house data (tabular) and you want to predict housing prices

So you can:

- Feature Creation – deriving new features like “price per square foot”
- Feature Selection – identifying and retaining important features such as location or number of bedrooms
- Feature Transformation – normalizing features to ensure they are on a similar scale, which helps algorithms like gradient descent converge faster

How does it look like on UNSTRUCTURED data:

For example you got a bunch of customer reviews

- Feature creation - converting text into numerical features using techniques like TF-IDF or word embeddings
- Feature creation - extracting image features such as edges or textures using techniques like convolutional neural networks (CNNs)

### Unsupervised learning

You basically want to discover inherent patterns, structures, or relationships within the input data. The machine uncovers the groups themselves. You gotta put the labels in the end.

Common techniques include:

#### Clustering

You group similar data points together into clusters based on their features.

for example, you got a bunch of customer data, like purchasing history, avg order value, prices etc. And you wanna tailor each group with specific marketing strategies.

You basically use **k-means clustering**

![unsupervised-clustering](./images/unsupervised-clustering.png)

#### Association Rule Learning

This is the classic example on figuring out how thing are related, like hey whoever buys bread also tends to buy butter. You basically identify associations between products.

Here you use the **apriori algorithm**.

The outcome is that then the store can place associated products together on the shelves to boost sales.

#### Anomaly Detection

This one is used a lot to detect credit card fraud. You basically identify transactions that deviate significantly from normal behavior.

Here you use **Isolation Forest** technique

Your outcome is a flag on potentially fraudulent transactions.

### semi-supervised learning

The best of both worlds kinda. You use a small amount of labeled data and a large amount of unlabeled data to train systems. Then the partially trained algorithm itself labels the unlabeled data.

The model is then re-trained on the resulting data mix without being explicitly programmed. So like... you give them a hint at the beginning and it does the rest.

![semi-supervised](./images/semi-supervised.png)

### self-supervised learning

You basically have a model generate pseudolabels for its own data without having humans label any data first. Then, using the pseudo labels, solve problems traditionally solved by Supervised Learning.

This is used a LOT on NLPs and image recognition tasks. This is mostly to teach models about what the data "represents".

Here's an example:
![self-supervised](./images/self-supervised.png)

### Reinforcement learning

RL is a type of Machine Learning where an agent learns to make decisions by performing actions in an environment to maximize cumulative rewards

It learns and learns from mistakes/successes. The main bits are:

- **Agent** – the learner or decision-maker
- **Environment** – the external system the agent interacts with
- **Action** – the choices made by the agent
- **Reward** – the feedback from the environment based on the agent’s actions
- State – the current situation of the environment
- Policy – the strategy the agent uses to determine actions based on the state

you use it to have AI play a game, for portfolio management, for autonomous vehicles, and other stuff like that.

The main idea is to maximize the cumulative reward over time.

Here's a good example on how it works
![reinforcement-learning](./images/reinforcement-learning.png)

### RLHF - reinforcement learning from human feedback

You want to add human feedback so that the model is more aligned with what a human what would choose.

It significantly improves model performance, like for LLMs its good for translations because it's actually like what a human would write, and not just a "technically correct" translation.

Basically you have the model create a response, and you separately have a bunch of human-generated responses.

And then the reward model is presented with both the syntethic and the human one. At some point it figures out what makes a human like a particular response.

A good summary is here:

![RLHF](./images/RLHF.png)

And the steps are just

1. Data collection (bunch of human prompts and responses)
2. supervised fine-tuning of a language model (you train your model and create responses that get compared to the human ones)
3. building a separate rewards model (humans indicate what response they liked best, and now this model knows about human preference)
4. optimize the language model with the reward-based model (You use this model as a reward fcn for the ML)

### Fit, bias, variance

Sometimes your model performance is poor. Causes include:

- overfitting : does great on training data, breaks down on test data.
- underfitting : does bad on training data, maybe model is too simple or data features are poor

You want a balanced data.

So you measure that level of balance by optimizing bias and variance.

- bias : the difference between predicted and actual data. **High bias == underfitting**. you're far from the truth. You can use a more complex model or increase the number of features.
- variance : How much the performance with another dataset of similar distribution. **High variance == overfitting**. To reduce you select important features, or splist data into test/train multiple times.

In short you want low variance and low bias:

![under-overfitting](./images/under-overfitting.png)

Here's a good visualization:

![bias-variance](./images/bias-variance.png)

### Model evaluation metrics

#### classification metrics

##### confusion matrix

You can evaluate how good your model was close to get to the truth by using a confusion matrix, and calculating the main 4 following metrics:

- precision : how precise were your predictions out of all the ones you flagged?
- recall : out of all the positives, how many did you correctly recall?
- F1 : combo of precision and recall
- accuracy : how many actual labels did you accurately predict? Only used for balanced dataset. Not great for spam/not spam because most is not spam i think.

![confusion-matrix](./images/confusion-matrix.png)

##### AUC-ROC

Another good one is the AUC-ROC. It's a curve that shows what the true positive compared to false positive looks like at various thresholds, with multiple confusion matrixes.

the idea is that the curve should get as close to being an inverted L.

![auc](./images/auc.png)

#### regression metrics

Some main ones are:

- MAE: mean absolute error
- MAPE: mean absolute percentage error
- RMSE: mean root square error
- R: variance of the model

These are all for continuous values.

### Inferencing

It is when the model makes a prediction made on new data.

So you can either have it

- real time : a chatbot that answers back at you right there and then
- batch : you analize a bunch of data at once ,you don't care about speed but about accuracy

you got two options at the edge:

- SLM : a small model stored in your phone or a local computer or something. Weaker model, but small latency.
- LLM: The model is huge, and it's stored on a server. Powerful model, but higher latency.

### Phases of ML project

It boils down to finding a problem to solve, getting data, figuring out what parts of the data matter, creating a model, checking it works, and deploying.

sorta like this:
![ML-phases](./images/ML-phases.png)

The key is to continuously monitor the outputs to keep the performance good and fix issues whenever they arize, whether you find drift or anything of hte sort.

TL;DR:

1. Define business goals - value, budget, success criteria, KPIs
2. ML problem framing - involve data scientists, determine if ML is appropriate
3. Data processing - data collection, integration, cleanup, formatting, and feature engineering
4. Model development - training, tuning, evaluation. It iterates and may need further add more feature engineering.
5. Retrain - look at data and features and see how to improve. Same with hyperparameters.
6. deployment - you select how to deploy (real-time, serverless, async, etc)
7. Monitoring - have a system track the performance. You detect early then you can fix stuff.
8. Iterations - model needs to be continously improved as more data comes in.

What about the data?
Need to do an exploratory data analysis before you do any feature engineering, so you know what is linked to what.

You can do a correlation matrix, so you can see how variables are linked to each other, and which features would be most important to the model.

### Hyperparameters

Fun fact, hyperparameters are the settings that define the model structure and algos.
You set it before you train.

Examples include:

- learning rate : how big the steps are when updating weighs. Higher == faster convergence, but you can overshoot.
- batch size : no of training samples used to update weighs.
- number of epochs : how many times the model will iterate over the entire training dataset. too few == underfitting. Too many == overfitting.
- regularization : adjusting balance between simple and complex models. More regularization == less overfitting.

You do hyperparameter tuning to optimize the model performance, so you get better accuracy and less overfitting.

How you do it?

- Grid search
- random search
- some automatic service such as sageMaker Automating Model Tuning (AMT)

Main issue, you got overfitting, what is going on?

- your model is good during training but bad with new data.
- training data was too small and did not represent the whole enchilada.
- maybe it trained for too long.
- maybe model was too complex and learned from noise.

How to fix:

- increase training datasize
- stop training earlier
- data augmentation (increase diversity in the data)
- adjust hyperparameters (can't really add more hyperparameters than you originally had. Also usually this is last resort.)

### When is ML not appropriate?

For **deterministic problems**, like when a problem can just be straight-up calculated, then jsut code it up.

If you use ML, like supervise/unsupervised/RL, then you're getting an approximation, never an exact value any time.

Even if LLMs have reasoning capabilities, they are imperfect and again just give you an approximation. So don't use LLMs for something you can just code.

### Udemy questions

1. You are trying to recognize handwritten digits. How do neural network function for this use case?

- Neural networks function like a decision tree algorithm and will create a set of rules
- neural networks will create a series of linear regression to identify numbers
- `neural networks will create several layers of interconnected nodes that will identify patterns in data --> this is literally what NNs do, all other options are not true to NNs`
- neural networks will create a database based on numbers and their pixel data

2. You are building a robot that learns how to cut vegetables. You are rewarding it for parallels and fine cuts. You want the actions to be safe and efficient. Which machine learning approach do you recommend?

- Supervised learning
- `Reinforcement learning --> you train the robot with rewards and penalties`
- unsupervised learning
- self-supervised learning

3. When a model is neither underfitting nor overfitting, it will have…

- `low bias and low variance -> the holy grail of good ML`
- high bias and high variance

4. Which AI application is used to automatically extract structured data from various types of documents, such as invoices, contracts, and forms?

- computer vision
- facial recognition
- fraud detection
- `intelligent document processing (IDP) -> its in the name`

5. You are building a model to predict the sale price of used cars, based on several attributes such as the brand, total number of kilometers, year of construction, etc… Which machine learning technique is appropriate for this task?

- clustering
- dimensionality reduction
- `regression --> mostly because price is a continuous variable`
- classification

6. What is the goal of feature engineering in the machine learning lifecycle?

- `to transform data and create variables for the model -> basically to make data useful for ML`
- to collect and clean data
- to calculate the model performance
- to maintain the desired level of performance

7. A company develops a machine learning model to predict stock prices. The model performs well on the training data but poorly on new, unseen data. Which option describes the fit of the model?

- it underfits --> you got high bias, meaning you're not good at predicting squat.
- it has low variance --> then it means its good, not the case
- `it overfits --> you have high variance, you got different answers across different sets of data, not good`
- it has high bias --> this is underfitting

## Section 9: AWS managed AI services

AWS AI services are pre-trained ML services, they are always available, they work across all regions, and perform well.

They are mostly pay-as-you-use

You also can setup a provisioned throughput if you already know what kinda workload you're expecting, for more savings.

Here's a bunch of the services:

![aws-ml-services](./images/aws-ml-services.png)

### Amazon comprehend

This is mainly for Natural Language Processing – NLP. It Uses **machine learning to find insights and relationships in text**. Like sentiment, keywords, language, etc.

Fully managed and serverless service. It automatically organizes a collection of text files by topic

Sample use cases:

- analyze customer interactions (emails) to find what leads to a positive or negative experience
- Create and groups articles by topics that Comprehend will uncover

you can use comprehend to classify docs using categories that you define. And of course, it supports a bunch of filetypes.

#### Named entity recognition NER

Comprehend can extract cool things like names, places, organizations, dates, etc.

So you can get from a bunch of text all the main bits.

You can also make a custom entity, as long as you give it examples that you can train this thing on.

#### TL;DR

**Can do custom classification and custom entity recognition**.

**Can do real-time AND async**.

You need at least 10 examples for custom classifications.

![comprehend](./images/comprehend.png)

### Amazon translate

It's a natural and accurate language translation.

Amazon Translate allows you to localize content (such as websites and applications) for international users, and to easily translate large volumes of text efficiently.

You can translate text directly, or a bunch of documents you upload or share via s3.

**Can do real-time AND async//batch**.

There are some metrics that amazon shows for translations. Like character count, time, throttle, etc.

You can also add your own terminology (domain specific words that you know how to translate)

parallel data--> basically you explain different ways to translate the same sentence based on context and tone. Friends vs office.

![translate](./images/translate.png)

### amazon transcribe

**Automatically convert speech to text**. Uses a deep learning process called automatic speech recognition (ASR) to convert speech to text quickly and accurately.

It can

- remove PII using 'redaction' (aws has like 5 options by default that you can redact, like names, phone no, ssn, etc)
- detect multiple languages

use cases include:

- transcribing client calls
- automating movie closed captions
- create metadata to enable searching on recordings.

to improve it you can:

- specify technical terms, acronyms, jargon, etc
- set **custom vocabulary**
- provide hints like pronunciation
- can add **custom language models** (you train the model on your own specific speech)

So instead of getting "my crow services" you actually get "microservices"

You can detect **toxicity** too. It uses ML on the background, and it looks at the tone and inflection of the way the person speaks. Same with the text info.

**It combines audio + text to describe toxicity**

![transcribe](./images/transcribe.png)

### amazon polly

The opposite of amazon transcribe. Now **you convert text to speech.** It uses deep learning to do so. this way you can have talking apps.

It has some cool advanced features

- lexicons: it can spell out acronyms, like AWS or WWW
- speech synthesis markup language (SSML): figures out how it should pronounce things, the cadence. like hello <pause> how are you?
- voice engine: can choose different types of voices
- speech mark: it knows where a sentence starts/ends in the audio. Good for lip syncing, or highlighting words.

![polly](./images/polly.png)

### amazon rekognition

You can recognize people, objects, texts, scenes, etc on images and videos using ML.

Basically it helps with:

- labeling
- text detection
- face recognition
- content moderation
- celebrity recognition
- detecting labels of companies and such
- path analysis (for recognizing players in games and stuff)

#### custom labels

You use this to identify your own logos from your companny.

You label training images, with your label and/or product. 100 images or less.

After that the ML figures out what your logo looks like and can detect it.

So you can see on social media if your logo is appearing on pictures and if that is good/bad for your brand.

#### content moderation

You only need 1-5% of human review, the rest is flagged by ML for harmful/bad content.

It's integrated with something called " Amazon augmented AI" for more human review.

So you can also use custom moderation adaptors. An image will pass/fail moderation, and when uncertain it can go for human review//amazon augmented ai (amazon A2I).

TL;DR, you can use rekognition to label your images and see if it passes/fails filters and return to user or not.

It's basically image analysis and detection. Can make your own custom case for your business. and you can train the recognition model by adding your own dataset.

![rekognition](./images/rekognition.png)

### amazon lex

you make chatbots quickly for your applications using voice and text.

Mostly for self-service bots. A one-stop shop for chatbot building.

It:

- supports multiple languages
- connects to all other aws stuff, like aws lambda, kendra, connect, comprehend, etc.
- it figures out the intent to call the correct lambda function to "fulfill the intent"
- the bot will ask for 'slots' (lambda input functions) that you need.

![lex](./images/lex.png)

You can create a bot in lex with either the **Traditional method** or the **Generative AI** method. For the latter you need bedrock.

On the traditional one, you can start blank, with an example bot, or with a bunch of transcripts.

You can add multiple language, you can select a voice, idle times, etc.

#### intent

For the bot you have to set up an intent. Which is, what will the user likely wnat to do, like. booking a hotel or a car, or some fallback for anything else.

To trigger an intent, you can specify what are likely phrases the user will say to trigger them.

And then you setup the slots, meaning the inputs you need at this intent to acutally call the lambda function.

#### visual builder

You can also figure out hte logic with their visual builder to see how the flow is looking like.

### amazon personalize

Its a service to build apps with real-time personalized recommendations.

this is what **amazon** uses to recommend items you'd be interested in, based on history, interactions, etc.

It is also really good at customized direct marketing.

Main bit is that this takes days to implement. And integrates with SMS and stuff like that.

![personalize](./images/personalize.png)

#### recipes

These are algorithms to use for specific use cases.

For example, you wnat to:

- recommend items to a user
- recommend trending items
- ranking items for a user
- getting similar items
- recommend next best action

All of this is recommending something for your user... personalized to their own choices.

### amazon textract

Extracting text from any scanned data using AI and ML.

Wide array of applications, should handle **handwriting**, various format documents, **forms and tables**, etc.

Amazon textract can extract all the raw text, and understand the layout of your page.

![textract](./images/textract.png)

It can also answer queries, analyze expenses (receipts), IDs, and other common types of documents.

### amazon kendra

**Its a document search service using ML**.

It extracts answers from a document (text, pdf, html, etc).

It basically **indexes all the data automatically** in the document.

Very good natural languages capabilities.

It can also incrementally learn and figure out preferred results.

Overall a search engine on steroids.

![kendra](./images/kendra.png)

TL;DR you wanna find something in the document? ask kendra.

### amazon mechanical turk

The idea is that you have **access to a distributed workforce**, where you get a bunch of humans do a lot of simple tasks.

Like image labeling.

The idea is the mechanical turk, was some "robot" playing chess that was actually controlled by some dude inside.

**used a lot for simple, easily distributed tasks**:

- image classificaiton
- data collection
- recommendation reviews

Deep integration with amazon A2I, sagemaker ground truth, etc.

Workers will work on those with a good reward. its a market.

![mechanical-turk](./images/mechanical-turk.png)

### amazon augmented AI (A2I)

your machine learning models are making predicitons in production, but you want humans to keep an eye on it.

**its human oversight of the ML predictions.**

so for high confidence predicitons, all is good, for low confidences, the results get sent to a human.

The reviewed data (with human in hte loop) can be used to further finetune the ML.

who can review?

- you employees
- contracted employees at AWS
- mechanical turk

some vendors are already pre-screened for security requirements.

The ML can be done in AWS or elsewhere like sagemaker, or rekognition or whatever.
![a2i](./images/a2i.png)

Its two steps:

1. you create a human review workflow (common ones are **textract** for key-value extraction, **rekognition** for image moderation, or some custom version)

- You can set a threshold for the confidence that would trigger a human review, or some percentage of the incoming data.

2. create and start a human in the loop

- Then you can set up a worker template, telling people what they should be looking for.
- set up who will review this, mechanical turk, or your private team, or a vendor in AWS marketplace

### amazon comprehend medical & transcribe medical

There is a version for amazon transcribe that is specifically geared for the medical space.

It is specialized because of HIPAA compliance.

It specializes in medical terminologies (medicines, conditions, etc).

you can either use the microphone or upload audios.

You can use "comprehend medical" to:

- detect useful information like notes, summaries, prescriptions, test results, etc
- figures out what is protected health information
- it stores it all in amazon s3
- you can do real-time analysis with _kinesis data firehose_.

basically you use transcribe --> convert to text --> analyze with comprehend

It figures out the relationships, the profile, and all that.

![comprehend-medical](./images/comprehend-medical.png)

### amazon's hardware for AI

**Amazon EC2 is the MOST POPULAR PRODUCT on amazon**.

EC2 = elastic computing cloud. It's infrastructure as service.

you can:

- rent virtual machines in the cloud (EC2)
- store data virtually (EBS)
- Distributing loads (ELB)
- scaling services (ASG)

you set up you're basically "building a computer":

- os
- computer power and cores
- storage
- ram
- network card
- firewall rules

to launch you can use a script from the get to.

Some instances are gpu-based (P3, P4, P5, ... G6...)

these ones are used a lot for ML//AI.

AWS went one level further and used something called "AWS trainium". Where it is using specific **chips built specifically for deep learning**.

- AWS trainium
  Trn I has I 6 trainium accelerators
  50% cost reduction when training a model

Also AWS inferentia, its a chip built to deliver inference at high performance and low cost.

- AWS inferentia
  4x throughput and 70% cost reduction
  INF2 and Inf2 are powered by inferentia.

These are more environmentally friendly because they use less electricity.

### Udemy questions

1. You should use Amazon Transcribe to turn text into lifelike speech using deep learning.

- true
- `false --> thats amazon polly`

2. A company would like to implement a chatbot that will convert speech-to-text and recognize the customers' intentions. What service should it use?

- transcribe
- rekognition
- connect
- `lex --> good one-stop-shop for making a chatbot`

3. You would like to find objects, people, text, or scenes in images and videos. What AWS service should you use?

- `rekognition --> this is specific for image recognition analysis`
- polly
- kendra
- lex

4. A start-up would like to rapidly create customized user experiences. Which AWS service can help?

- `personalize --> it litearlly is what amazon.com uses, so`
- kendra
- connect

5. A research team would like to group articles by topics using Natural Language Processing (NLP). Which service should they use?

- Translate
- `Comprehend --> its literally meant to comprehend whats in the text and automatically organizes them in groups`
- Lex
- Rekognition

6. A company would like to convert its documents into different languages, with natural and accurate wording. What should they use?

- Transcribe
- polly
- `translate --> simple name, it does what its named after`
- wordTranslator

7. Which AWS service makes it easy to convert speech-to-text?

- connect
- translate
- `transcribe --> like the name, it just transcribes speech to text`
- polly

8. Which of the following services is a document search service powered by machine learning?

- Translate
- `Kendra --> ask kendra!! powerful search indexing engine of sorts`
- Comprehend
- Polly

9. Which AWS service enables human reviews of ML predictions?

- Amazon sagemaker jumpstart
- Amazon kendra
- `Amazon augmented AI (Amazon A2I) --> a2i is to set up human reviews in the process`
- AWS deepracer

## Section 10: amazon Sagemaker (AI) - deep dive

Its a fully managed AWS service to create an ML model.

Usually it is difficult to do all the ML dev steps + provisioning deployment servers

it's an end-to-end service:
![sagemaker](./images/sagemaker.png)

It has a ton of built-in algorithms that would otherwise have been inported in a library, such as:

- supervised algos:
  - linear regression and classification
  - KNN algorithms (for classification)
- unsupervised algos:
  - PCA
  - K-means
  - anomaly detection
- image processing: classificaiton, detection, etc
- textual algos: NLP, summarization

### Automatic model tuning AMT

You define the objective metric, and AMT figures out the best hyperparameters ranges, search strategy, max runtime and stop conditions.

It saves you time and money because it will stop things if they don't work out.

### Model deployment and inference

To deploy you got four options:

- real-time: one prediction at a time
- serverless: idle periods between traffic spikes, can tolerate cold starts
- asynchronous: you send big data up to 1GB, you get long processing times, and you need near-real time latency. Need to stage data on s3
- batch: multiple predictions at once (entire dataset). Need to stage data on S3

Here's a summary:

![sagemaker-comparison](./images/sagemaker-comparison.png)

You tend to work with sagemaker studio, which is like a UI you can use as a team. Seems that you can use jupyter notebooks and flat-out code on this UI.

Sagemaker studio is an end-to-end ML development from a unified interface, so you do development all the way to deployment and automation.

### Data wrangler

This interface is for data... wrangling

From data exploration, to cleanup, extraction, preparation, feature engineering, etc.

It has SQL support

It has a data quality tool to figure out datatypes and what you have missing.

![sagemaker-wrangler](./images/sagemaker-wrangler.png)

#### What are ML features?

**Features are inputs to ML models used during training and used for inference.**

You need to have high-quality features in all your datasets for you to reuse.

You can use **Feature Store** to ingest data from various sources and create your features automatically as needed. You can publish these features directly from the Wrangler into the Feature Store, and find them on Sagemaker Studio.

### Clarify

this one evaluates foundation models, like the friendliness or humor of the model.

You can use humans to evaluate the foundation model(s). Be it an AWS managed team or your own people.

You can use built-in datasets or your own. and you can use built-in metrics and algorithms.

![sagemaker-clarify](./images/sagemaker-clarify.png)

#### Explainability

You can use sagemaker clarify (its a set of tools) to understand how/why the ML model is making some predictions.

Basically you study the cause/effect (Explainability) of the ml model inputs and the predicted output. It should give an idea of the effect of each input parameter.

You can also use it to debug once it's deployed.

So basically you use this one to answer questions like:

- “Why did the model predict a negative outcome such as a loan rejection for a given applicant?”
- “Why did the model make an incorrect prediction?”

#### Human bias

We want to detect and explain biases in our models. So we measure that using statistics, and some default set of inputs.

#### Ground truth

- RLHF= reinforcement learning from human feedback

We want to align a model to human preferences and so we add the 'human feedback' on the reward function

- Human feedback for ML
  You literally do data annotation or evaluation by hand.
  you create your own labels.

Reviewers can be: AWS mechanical turk workers, your employees, 3rd party vendors

**Sagemaker ground truth plus = label data**

### ML governance

Sagemaker helps with:

- Model cards: essential model info, intended use, risk, training details
- model dashboard: centralized repo with all info of all models
- role manager: defines roles for people, MLOps, data scientist, etc

On the **model dashboard**, expect mlflow stuff. you can track which models are deployed for inference. It helps you find models that violate thresholds you set for data quality, model quality, bias, explainability

On the **model monitor**, you literally monitor the quality of your model, be it continously or on a schedule. It alers when stuff drifted, so you can fix it.

On the **model registry**, you basically have a repo to track, manage, and version ML models. you can **manage the approval status of a model**, automate model deployment, share models, etc.

On the **pipelines**, you automate the process of building, training and deploying an ML model. IT's CI/CD. And it lets you automatically train 100s of models.

For the pipelines, there is an established procedure:

![sagemaker-pipelines](./images/sagemaker-pipelines.png)
