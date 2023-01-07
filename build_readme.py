import os
from subprocess import check_output
import re

# Metadata
APPLICATION_PATH = os.path.join(os.getcwd(),"application.py")
README_PATH = os.path.join(os.getcwd(),"README.md")
VERSION_PATH = os.path.join(os.getcwd(),"VERSION.txt")
README_META_PATH = os.path.join(os.getcwd(),"readme_content.txt")

# Paths
GENERATOR_PATH = os.path.join("src","generators")
SDF_GENERATION_PATH = os.path.join("data","generated")

def get_generator_help_text():
    help_texts = {}
   
    for entry in os.listdir(GENERATOR_PATH):
        if ".py" in entry and entry != "base.py":
            p = check_output(["python",f"{os.path.join(GENERATOR_PATH,entry)}","--help"])
            help_texts[f"{entry}"] = bytes.decode(p,"utf-8").rstrip()

    return help_texts

def write_help_texts(content:str):
    help_text_content = ""
    for key,value in get_generator_help_text().items():
        help_text_content += f"**Generator-Source: {key}**\n```\n{value}\n```\n\n"
    
    content = content.replace("<GENERATOR_HELP_TEXTS>",help_text_content)
    return content

def get_default_app_args():
    EXAMPLE_APPARGS = ""
    record = False
    with open(APPLICATION_PATH,"r") as appfile:
        for line in appfile.readlines():
            if "APP_ARGS START" in line:        
                record = True
                continue
            elif "APP_ARGS STOP" in line:
                record = False
                break
            if record == True:
                EXAMPLE_APPARGS += line
    return EXAMPLE_APPARGS

def get_version():
    with open(VERSION_PATH,"r") as versionfile:
        return versionfile.readlines()[0]

def get_paths():
    return {
        "<GENERATOR_LOCATION>": str(GENERATOR_PATH),
        "<GENERATED_JSON_LOCATION>": str(SDF_GENERATION_PATH),
    }

def get_meta_readme():
    content = ""
    with open(README_META_PATH,"r") as contentfile:
        for line in contentfile.readlines():
            content += line
    return content

def replace_meta_content(version:str, appargs:str,content:str):
    content = content.replace("<VERSION>",version)
    content = content.replace("<DEFAULT_ARGS>",appargs)
    return content

def replace_paths(content:str,**kwargs):
    for key,value in kwargs["paths"].items():
        try:
            content = content.replace(key,value)
        except:
            raise ValueError(f"Could not rebuild 'README.md'. Placeholder {key} was not found in 'readme_content.txt' at location {str(README_META_PATH)}")
    return content

def make_readme(content:str):
    if os.path.exists(README_PATH):
        os.remove(README_PATH)

    with open(README_PATH,"w") as readmefile:
        readmefile.write(content)

def replace_placeholders(**kwargs):
    content = get_meta_readme()
    content = replace_meta_content(kwargs["version"],kwargs["appargs"],content)
    content = replace_paths(content,paths=kwargs["paths"])
    content = write_help_texts(content)
    return content

def run():
    make_readme(replace_placeholders(
        version=get_version(),
        appargs=get_default_app_args(),
        paths=get_paths(),
    ))
    print("Recreated README.md")

if __name__ == "__main__":
    workdir = os.getcwd()
    assert workdir.split("\\")[::-1][0] == "diffconvpl"
    run()