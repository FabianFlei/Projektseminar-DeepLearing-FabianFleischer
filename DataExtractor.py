import numpy as np

"""Noch hinzufügen MAX Value"""

# Als Grundlageninformation https://www.python-lernen.de/dateien-auslesen.htm
class dataExtractor():
    
    def extractData(dataSet = None):
        metadata = {"dataSet": ["a280.tsp", "brd14051.tsp", "berlin52.tsp"]}

        assert dataSet is None or dataSet in metadata["dataSet"]


        data = open(dataSet, 'r')

        # Formatiere Name
        name = str(data.readline())
        find = name.find(": ")
        if find != -1:
            name = name[find+2:len(name)-1]
        else: 
            name = name[name.find(":")+1:len(name)-1]

        # Formatiere Comment
        comment = str(data.readline())
        find = comment.find(": ")
        if find != -1:
            comment = comment[find+2:len(comment)-1]
        else: 
            comment = comment[comment.find(":")+1:len(comment)-1]
        
        # Formatiere Type
        type = str(data.readline())
        find = type.find(": ")
        if find != -1:
            type = type[find+2:len(type)-1]
        else: 
            type = type[type.find(":")+1:len(type)-1]

        # Formatiere Dimension   
        dimension = str(data.readline())
        find = dimension.find(": ")
        if find != -1:
            dimension = int(float(dimension[find+2:len(dimension)-1]))
        else: 
            dimension = int(float(dimension[dimension.find(":")+1:len(dimension)-1]))

        # Formatiere Edge_weight_type
        edge_weight_type = str(data.readline())
        find = edge_weight_type.find(": ")
        if find != -1:
            edge_weight_type = edge_weight_type[find+2:len(edge_weight_type)-1]
        else: 
            edge_weight_type = edge_weight_type[edge_weight_type.find(":")+1:len(edge_weight_type)-1]

        # Formatiere Node_coord_section
        node_coord_section = str(data.readline())
        node_coord_section = node_coord_section[:len(node_coord_section)-1]

        maxValue = 9999999999999999999

        maxX = 0
        minX = maxValue

        maxY = 0
        minY = maxValue

        target_location = []

        while True:
            dataPoints = data.readline()
            # Fall ist EOF erreicht == Alle Daten eingelesen
            if dataPoints.find("EOF") != -1:
                break


            index = int(str(dataPoints).index(" "))
            while index == 0:
                dataPoints = dataPoints[1:]
                index = int(str(dataPoints).index(" "))

            dataPoints = dataPoints[index:]
            index = 0

            while index == 0:
                dataPoints = dataPoints[1:]
                index = int(str(dataPoints).index(" "))

            x = int(float(dataPoints[:int(str(dataPoints).index(" "))]))

            if x > maxX:
                maxX = x
            elif x < minX:
                minX = x


            dataPoints = dataPoints[index:]
            find = 0

            while find != -1 and find == 0:
                dataPoints = dataPoints[1:]
                find = int(str(dataPoints).find(" ")) 

            if find != -1:
                y = int(float(dataPoints[:len(dataPoints)-1])) # vor \n am ende jeder Zeile ist noch ein Leerzeichen
            else:
                y = int(float(dataPoints[:len(dataPoints)-1])) # -2 da \n am ende jeder Zeile

            if y > maxY:
                maxY = y
            elif y < minY:
                minY = y


            target_location.append(np.array([x,y]))

        if minX > minY:
            minValue = minY
        else:
            minValue = minX

        if maxX > maxY:
            maxValue = maxX
        else:
            maxValue = maxY

        offset = minValue-1

        divident = (maxValue) / 200

        for i in range(0,len(target_location)):
            target_location[i] = np.round( (target_location[i]) / divident )


        info = {
                "name": name, 
                "comment": comment, 
                "type": type, 
                "dimension": dimension,
                "edge_weight_type": edge_weight_type, 
                "node_coord_section": node_coord_section,
                "minValue": minValue,
                "maxValue": maxValue,
                "offset": offset,
                }
        
        target_location

        print(target_location)

        return target_location, info


