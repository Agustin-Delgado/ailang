# justify

Fully justify text. The first line of standard input is the line width W (W >= 1). The rest is text:
paragraphs are separated by one or more blank lines (lines empty or only spaces); within a paragraph,
words are separated by any spaces and line breaks.

Lay out each paragraph greedily: put as many words on a line as fit with at least one space between
them. Every line except the paragraph's last is then padded to exactly W characters by widening the
gaps between words: spread the extra spaces as evenly as possible, giving the leftmost gaps one more
when they do not divide evenly. A line with a single word, and the paragraph's last line, are left
as they are (words separated by single spaces, no trailing spaces). A word longer than W goes on a
line of its own.

Print the paragraphs separated by one empty line. Print nothing if there are no words.
