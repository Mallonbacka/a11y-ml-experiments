# Describing Trends with GPT-3

This Markdown version is included becuase GitHub doesn't render the block outputs automatically sometimes - here you can see how the API responded without needing an account of your own. 

## The Scenario

Let's imagine we need to graph pageviews over seven days.

For our sighted users, a simple line plot (days on the x-axis, pageviews on the y-axis) is a quick way to get a sense of what's been going on. Sighted users see an overall trend quickly, then can dig into the numbers some other way (table?) if they want to know more. The numbers are dynamic, coming directly from a database. There are lots of tools to programatically generate charts from data.

But how do we cater for blind users? We need an `alt` attribute for the chart image. One option is to just list the values, but what if they are huge numbers? Imagine a screen reader saying "Monday three hundred and twenty one thousand eight hundred and seventy nine, Tuesday ninety seven thousand two hundred and twelve..." - it's hard to appreciate the trend, and it's an overwhelming amount of information and not useful in the way that an overall trend would be.

Even showing just one week, we have relatively few numbers, but it's still going to take a lot of logic to create a big nested `if...else` block to describe every possible trend manually. And how will we handle relative changes?

After all of the hype in the last few years, let's experiment with the GPT-3 model from OpenAI, and see how it _might_ solve this problem.

## Getting set up

I've started by running `pip install openai` to grab the Python bindings. 


```python
import os # We only need this to get our API key from the environment variable.
import json
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")
```

And a quick test that `openai.Completion.create` works:


```python
response = openai.Completion.create(engine="text-davinci-002", prompt="Say this is a test", temperature=0, max_tokens=6)
print(response)
```

    {
      "choices": [
        {
          "finish_reason": "length",
          "index": 0,
          "logprobs": null,
          "text": "\n\nThis is a test"
        }
      ],
      "created": 1648109397,
      "id": "cmpl-4pCpZBRGMXui3G1jJKJRo62DsOK2y",
      "model": "text-davinci:002",
      "object": "text_completion"
    }


Looking good. 

There's only one `choice`, and we can see the `text` for that choice, it says `\n\nThis is a text"`.

In future blocks, let's just print the part we are interested in.

## The Prompt

GPT-3 works with "completions" - we write a "prompt", and it continues with what it things is most likely to come next. 

You can find [some examples of completions](https://beta.openai.com/examples) on OpenAI's website. 

For this example, let's try something like "summarize the trend of the list of numbers [numbers] in one sentence". I'll also set `max_tokens` to 36 (maybe about 25 words) plenty for a reasonable length sentence.


```python
response = openai.Completion.create(engine="text-davinci-002", 
                                    prompt="Summarize the trend of the list of numbers 132, 329, 583, 743, 966, 1123, 1298 in one sentence", 
                                    temperature=0, 
                                    max_tokens=36)
print(response["choices"][0]["text"])
```

    
    
    The list of numbers is increasing.


Not a bad start, they definitely are increasing, but we're still missing a lot of potentially useful information - what are they increasing from and to? What is the list of numbers?

GPT-3 certainly recognises the concepts here, which is impressive, but it could be better.

## Show some examples

Let's try showing some examples of what we'd like to see. Let's use these numbers as one example, then let's try another example with a different type of trend:


```python
response = openai.Completion.create(engine="text-davinci-002", 
                                    prompt="""Summarize the trend of the list of numbers in one sentence

Numbers: 132, 329, 583, 743, 966, 1123, 1298
Trend: The numbers increase steadily from just over 100 to almost 1300.

Numbers: 300, 323, 293, 313, 341, 301, 329
Trend: The numbers fluctuate between 293 and 341.

Numbers: 99, 97, 96, 90, 87, 83, 82
Trend: 
""", 
                                    temperature=0, 
                                    max_tokens=36)
print(response["choices"][0]["text"])
```

    
    The numbers decrease steadily from just under 100 to just under 83.


Slightly odd choice of wording here, but this is cool!

Let's see how it handles a trend not covered by the examples at all:


```python
response = openai.Completion.create(engine="text-davinci-002", 
                                    prompt="""Summarize the trend of the list of numbers in one sentence

Numbers: 132, 329, 583, 743, 966, 1123, 1298
Trend: The numbers increase steadily from just over 100 to almost 1300.

Numbers: 300, 323, 293, 313, 341, 301, 329
Trend: The numbers fluctuate between 293 and 341.

Numbers: 55, 54, 57, 5643, 56, 55, 54
Trend: 
""", 
                                    temperature=0, 
                                    max_tokens=36)
print(response["choices"][0]["text"])
```

    
    The numbers fluctuate between 54 and 57, with one outlier at 5643.


Wow. We've got the concept of an outlier without having to include it in an example.

Next, let's try adding the concept of time. We said we're dealing with one week, so let's add references to days of the week to see if we can get _when_ the outlier occurred included.


```python
response = openai.Completion.create(engine="text-davinci-002", 
                                    prompt="""Summarize the trend of the list of numbers in one sentence

Numbers: 132, 329, 583, 743, 966, 1123, 1298
Trend: The numbers increase steadily from just over 100 on Monday to almost 1300 on Sunday.

Numbers: 300, 323, 293, 313, 341, 301, 329
Trend: The numbers fluctuate between 293 and 341.

Numbers: 55, 54, 57, 5643, 56, 55, 54
Trend: 
""", 
                                    temperature=0, 
                                    max_tokens=36)
print(response["choices"][0]["text"])
```

    
    The numbers fluctuate between 54 and 57.


This tiny tweak in one of the examples completely hides the existance of the outlier. 

Now let's add an outlier example, but at a different scale...


```python
response = openai.Completion.create(engine="text-davinci-002", 
                                    prompt="""Summarize the trend of the list of numbers in one sentence

Numbers: 132, 329, 583, 743, 966, 1123, 1298
Trend: The numbers increase steadily from just over 100 on Monday to almost 1300 on Sunday.

Numbers: 300, 323, 293, 313, 341, 301, 329
Trend: The numbers fluctuate between 293 and 341 all week.

Numbers: 7, 8, 11, 9, 218, 8, 8
Trend: The numbers fluctuate between 7 and 11 all week, wih one outlier at 218 on Friday.

Numbers: 55, 54, 57, 5643, 56, 55, 54
Trend: 
""", 
                                    temperature=0, 
                                    max_tokens=36)
print(response["choices"][0]["text"])
```

    
    The numbers fluctuate between 54 and 57 all week, with one outlier at 5643 on Wednesday.


Now it's just wrong! It's got the right idea, but the day of the week is wrong.

Let's see if being extra-clear in the examples helps...


```python
response = openai.Completion.create(engine="text-davinci-002", 
                         prompt="""Summarize the trend of the daily list of numbers in one sentence

Numbers: M: 132, T: 329, W: 583, T: 743, F: 966, S: 1123, S: 1298
Trend: The numbers increase steadily from just over 100 on Monday to almost 1300 on Sunday.

Numbers: M: 300, T: 323, W: 293, T: 313, F: 341, S: 301, S: 329
Trend: The numbers fluctuate between 293 and 341 all week.

Numbers: M: 7, T: 8, W: 11, T: 9, F: 218, S: 8, S: 8
Trend: The numbers fluctuate between 7 and 11 all week, wih one outlier at 218 on Friday.

Numbers: M: 55, T: 54, W: 57, T: 5643, F: 56, S: 55, S: 54
Trend: 
""", 
                         temperature=0, 
                         max_tokens=36)
print(response["choices"][0]["text"])
```

    
    The numbers fluctuate between 54 and 57 all week, with one outlier at 5643 on Thursday.


That helped! Let's check if a two-day dip works too...


```python
response = openai.Completion.create(engine="text-davinci-002", 
                         prompt="""Summarize the trend of the daily list of numbers in one sentence

Numbers: M: 132, T: 329, W: 583, T: 743, F: 966, S: 1123, S: 1298
Trend: The numbers increase steadily from just over 100 on Monday to almost 1300 on Sunday.

Numbers: M: 300, T: 323, W: 293, T: 313, F: 341, S: 301, S: 329
Trend: The numbers fluctuate between 293 and 341 all week.

Numbers: M: 7, T: 8, W: 11, T: 9, F: 218, S: 8, S: 8
Trend: The numbers fluctuate between 7 and 11 all week, wih one outlier at 218 on Friday.

Numbers: M: 37465, T: 52374, W: 39809, T: 30885, F: 44325, S: 230, S: 223
Trend: 
""", 
                         temperature=0, 
                         max_tokens=36)
print(response["choices"][0]["text"])
```

    
    The numbers increase steadily from just over 37,000 on Monday to almost 45,000 on Friday, with a sharp decrease on Saturday and Sunday.


This is OK - it's got the two days correctly identified and uses the word "sharp" correctly, but it gives some slightly misleading information. It says that the numbers "increase steadily", but they actually jump around a little, crossing 52K on Tuesday. 

The solution is definitely more examples, but we are moving into the territory where it might make more sense to use the API's [fine-tuning options](https://beta.openai.com/docs/guides/fine-tuning). If we don't, our prompt is getting bigger and bigger, and we will be billed (once we run out of free credit) for sending the prompt with every request. 

## Next Steps?

1. Try a fine-tuned model, as described above
2. Collect lots of examples - maybe these could be semi-automated with Mechanical Turk or a similar service?
3. Explore what happens with different time intervals - can we pass dates instead of days?

