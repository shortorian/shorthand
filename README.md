# shorthand
`shorthand` is a Python package that allows a user to parse text data according to custom syntax files.

The original intent was to create custom data entry formats that enable rapid manual entry of text data tailored to specific applications, such as transcribing citation data from large sets of scientific papers. The result is a kind of meta-language that allows (nearly) arbitrarily complex linking between pairs of strings and substrings in a CSV file. With a well-crafted syntax, users can write easily readable lines that can be parsed into tens of rows of tabular data suitable for bulk input into normalized databases. Each line of a `shorthand` input file is a comma-separated pair of strings structured according to the syntax files, so syntaxes that define input values which are tab-completable in a text editor can allow users to manually generate thousands of lines of tabular data very quickly.

`shorthand` is in early development, with an alpha release and more detailed documentation expected in summer 2022.
