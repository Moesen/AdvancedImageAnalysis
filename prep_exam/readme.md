## Commands for finding stuff in pdf
Use pdfgrep to find them

`pdfgrep PATTERN FILE`
- **Examples**
`pdfgrep` 'laplacian' lecture\_notes/\*pdf

Usefull patterns
- -i: ignore case
- -c: count
- -p: page count
- -l: files with match 
- -L: files without match
- -H: with filename
- -h: without filename
- --exclude-GLOB: Exclude certain files
