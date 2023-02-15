import stanfordnlp
from nltk.tokenize import word_tokenize , sent_tokenize

# To read text file
print("Hello , my name is Hossein Simchi")
f = open("C:\\Users\\Lenovo\\Desktop\\dataset.txt" , encoding = "UTF-8")
y = f.read()


#print original size
print("print original size : {}".format(len(y)))

# To Remove unwanted character and to clean text file
import re
from nltk.corpus import wordnet

class RepeatReplacer(object):
    def __init__(self):
        self.repeat_regexp=re.compile(r'(\w*)(\w)\2(\w*)')
        self.repl=r'\1\2\3'
    def replace(self,word):
        if(wordnet.synsets(word)):
            return word
        repl_word=self.repeat_regexp.sub(self.repl,word)
        if(repl_word!=word):
            return self.replace(repl_word)
        else:
            return repl_word
replacer=RepeatReplacer()
replacer.replace(y)
unwanted_alpha=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
for alpha in unwanted_alpha:
    y=y.replace(alpha,"")
unwanted_digit=['0','1','2','3','4','5','6','7','8','9']
for digit in unwanted_digit:
    y=y.replace(digit,"")
unwanted_punc=['"',"'",'=','@','&','%','.',',',':','\\','$','^','<','>','!','?','{','}',';','\n','\t','(',')','[',']','/','*','+','#','\u200c','\ufeff','-','_','|']
for punc in unwanted_punc:
    y=y.replace(punc,"")

# To print size after Remove extra characters
print("print size after Remove extra characters is {} ".format(len(y)))

# Tag sentences depends om ms.seraji corpus


#stanfordnlp.download('fa', MODELS_DIR, False) it's enough for the first time
# We open the clean text that acheived in the above code
t = open ("C:\\Users\\Lenovo\\Desktop\\clean_text.txt" , encoding = "UTF-8")
file = t.read()
MODELS_DIR = r'C:\Data'
nlp = stanfordnlp.Pipeline(lang="fa", models_dir=MODELS_DIR, treebank='fa_seraji', use_gpu=False)
doc = nlp(file)
print("print POS tag : ")
print(*[f'text: {word.text+" "}\tupos: {word.upos}\txpos: {word.xpos}' for sent in doc.sentences for word in sent.words], sep='\n')




