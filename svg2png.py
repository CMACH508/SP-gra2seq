#   Copyright 2022 Sicong Zang
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
#   P.S. We thank Guoyao Su, Yonggang Qi and et al. for the codes of sketch cropping
#        in https://github.com/sgybupt/SketchHealer.
#
""" Convert .SVG file to .PNG file """

import cairosvg
import os

def exportsvg(fromDir, targetDir, exportType):
    num = 0
    for a, f, c in os.walk(fromDir):
        for fileName in c:
            path = os.path.join(a, fileName)
            if os.path.isfile(path) and fileName[-3:] == "svg":
                num += 1
                fileHandle = open(path)
                svg = fileHandle.read()
                fileHandle.close()
                exportPath = os.path.join(targetDir, fileName[:-3] + exportType)
                exportFileHandle = open(exportPath, 'w')
                if exportType == "png":
                    try:
                        cairosvg.svg2png(bytestring=svg, write_to=exportPath)
                    except:
                        print("error in convert svg file : %s to png." % path)
                exportFileHandle.close()
                print("Success Export ", exportType, " -> ", exportPath)

# svgDir = './sample/'  # Directory of targets (.svg)
# exportDir = './sample/'  # Directory of outputs (.png)
# if not os.path.exists(exportDir):
#     os.mkdir(exportDir)
# exportsvg(svgDir, exportDir, 'png')
