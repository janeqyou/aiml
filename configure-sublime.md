## Sublime Text 3 + Markdown + MathJax

### Install the latest sublime text 3
* The following instructions only apply to sublime text 3
### Install MarkdownPreview and MarkdownEditing 
* Tool->Command Palette( command + shift + p )->Install Package Control
* Command Palette -> "install". Hit return and type MarkdownPreview in the poped up input bar. If MarkdownPreview is succesfully installed, a file named "Package Control Message" will show in a new tab.
* Similarly, install MarkdownEditing package. 
* Restart your Sublime Text and open a ".md" file see the changes. 

### Update key binding to generate markdown preview 
* Preference -> Key Bindings. Add `{ "keys": ["alt+m"], "command": "markdown_preview", "args": {"target": "browser", "parser":"markdown"} }`to the right window. Close the windows
* Go to the ".md" file and Alt+m. A browser should pop up and show the preview of Markdown

### Install and configure MathJax
* Preference -> Package Settings -> MarkdownPreview -> settings 
* Add the following snippet to the right window and close <br>
`{
    "enable_mathjax": true,
    "js": [  
    "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js",  
            "res://MarkdownPreview/js/math_config.js",  
    ],  
    "markdown_extensions": {  
        "pymdownx.arithmatex": {  
            "generic": true  
        }  
    }  
}`
* Go to ".md" file and add 
$a^2 + b^2 = c^2$

### Resources
* [Install markdown for sublime text](http://plaintext-productivity.net/2-04-how-to-set-up-sublime-text-for-markdown-editing.html)
* [Enable mathjax for sublime text](https://stackoverflow.com/questions/27972177/how-to-enable-mathjax-rendering-in-sublime-text-markdown-preview)
