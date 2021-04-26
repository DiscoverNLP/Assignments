---
# Lab 1: Regular Expressions
---

## RegEx Practice (15 points)

Practice regular expressions by playing [RegEx Golf](https://alf.nu/RegexGolf).

You *must* complete the following (14 points, autograded): 

* Warmup (2 pts)
* Anchors (2 pts)
* It never ends (2 pts)
* Ranges (2 pts)
* Backrefs (2 pts)
* One additional puzzle of your choosing (4 pts for the first puzzle, 1 pt extra credit for each additional solution)

### Turn In

In the file `golf.py`, fill in the empty strings with your best (that is, shortest) regular expression for each of the puzzles that you solved. You'll get 1 point for turning in a nontrivial file, and full points for each solution as long as your regular expression didn't avoid the challenge of writing rules that generalize properties of the words (e.g., `r"^(word1|word2|word3...)$"` would be ridiculously long and a little silly). That said, if your length seems multiple times longer than what's on the leaderboards, challenge yourself to get one that's shorter!

### Integrity Note

Since these puzzles are online, there are (undoubtedly) solutions posted somewhere. I trust you not to look for those solutions, but to submit the best solutions you can come up with on your own. 

---

## Chatbot (35 points)
We’ll use Discord for communicating with each other this semester. There will be channels set up on our Discord server for each of the assignments; additionally, you should feel free to create your own channels for communicating with your teammates. Stop now and make sure that you have access to the CS159 Discord Server.

For this part of lab, you will make an ELIZA-like chatbot that can join our Discord server. You will follow these steps:

1. Set up a Discord bot user with the instructions [here](https://sites.google.com/g.hmc.edu/cs159spring2021/labs/lab-1-regular-expressions/discord-bot-setup). 
2. Open the file discordbot.py and paste in your access token.
3. Start your bot by running `python3 discordbot.py`. To see the bot in the Discord app, look in the right-hand panel for the username of your bot.

Once you have those steps working, you’re ready to improve upon your bot.

The only requirement is that you must modify the `make_reply` function in `discordbot.py` to respond to messages from Discord users. Your function must make use of regular expressions using the Python [re](https://docs.python.org/3/library/re.html) library. In particular, you should use each of the following at least once (3 points):

* Regular expression groups ((...)). *Note*: these should be used to capture content that your chatbot processes later; don't just put parentheses around a regular expression and then ignore the group.
* Character classes ([...], \d, \D, \w, \W, \s, \S, etc.)
* Regular expression quantifiers (+, *, and/or {...})

You can choose the domain and personality of your both. It can be a Rogerian psychologist like ELIZA, or you can choose a different domain. If you choose a different domain, make sure it’s one where you can reasonably expect repetition, so regular expressions will be effective.

On Gradescope, you should submit your completed `discordbot.py` file with your implementation. (1 point for a file with any changes)
You should also submit a .pdf report that describes your Bot. That report should have the following sections:

* **Overview**: An introduction to your DiscordBot. What domain does it work in? What is its personality? (5 points for completeness and clarity)
* **Regex Description**: Clearly describe how you make use of each of the required regular expression features in your make_message function. (5 points for completeness and clarity)
* **Analysis**: Describe at least one interaction with your bot that worked well, and at least one interaction with your bot that works poorly (or not as one might expect). You should include a screenshot or transcript of your Discord conversations in your writeup. This should include explanation not just of what happened, but why it happened, and why that was good or bad. (12 points for completeness and clarity)
* **Future Directions**: Thoughtfully describe how you would address the existing shortcomings of your bot if you had more time. For example, what would you do to make your bot better if you had another week to work on it? If you were going to use this code as the starting point for a final class project? (8 points for completeness and clarity)

The starter code has a Markdown file that you can use to write your analysis. To turn your markdown into a `.pdf` file, you should use pandoc:

```
pandoc analysis.md -o analysis.pdf
```

If you're having trouble getting pandoc installed and running on your own machine, you can refer to the Docker instructions for a way to use Docker to do this. Please add your `analysis.pdf` file to your GitHub repository so that we can view it for grading. (1 point for having a nonempty analysis file)
