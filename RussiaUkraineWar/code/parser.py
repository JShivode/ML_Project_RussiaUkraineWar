import csv

def build_data():
    labels = get_features()  # get features data and assign it to the variable labels
    label_personnel = get_feature_personnel() # get feature personnel data and assign it to the variable label_personnel
    average_temp = get_average_temperature()  # get average temperature data and assign it to the variable average_temp


    export_csv =[]
    # iterate over the zip of labels, label_personnel, and average_temp keys
    for label_date,label_personnel__date ,whether_date in zip(labels,label_personnel,average_temp.keys()):
        row_csv = {
                'date': whether_date, # assign the value of whether_date to the key 'date'
                'personnel' : label_personnel[label_personnel__date]
            }
        row_csv.update(average_temp[label_date])  # update row_csv with the data from average_temp at the key label_date
        row_csv.update(labels[label_date])
        export_csv.append(row_csv) # append row_csv to export_csv

    rows_info = ['date', 'temp', 'personnel', 'aircraft', 'helicopter','tank','APC', 'field_artillery', 'MRL','drone']
    # create a new CSV file called data_effect_on_fighting.csv, and write the rows_info and export_csv data to it
    with open('../data/data_effect_on_fighting.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames = rows_info) # create a DictWriter object that writes to csvfile and uses rows_info as the fieldnames
        writer.writeheader() # write the header to the CSV file
        writer.writerows(export_csv) # write the data from export_csv to the CSV file

def get_features():
    result:dict =dict()  # create an empty dictionary called result
    with open('../data/russia_losses_equipment.csv', newline='') as csvfile:  # open the file russia_losses_equipment.csv
        features = csv.DictReader(csvfile)  # read the CSV file and create a DictReader object called features
        aircraft_temp=0 # set aircraft_temp to 0
        helicopter_temp=0
        tank_temp=0
        APC_temp=0
        field_artillery_temp=0
        MRL_temp=0
        drone_temp=0
        for row_feature in features: # iterate over each row in the CSV file
            # add a new entry to the result dictionary, with the date as the key and a dictionary of equipment losses as the value
            result.update({row_feature['date']: {
                'aircraft' : (float(row_feature['aircraft']) - float(aircraft_temp)), # calculate the difference in aircraft losses since the last row
                'helicopter' : (float(row_feature['helicopter']) - float(helicopter_temp)),
                'tank' : (float(row_feature['tank']) - float(tank_temp)),
                'APC' : (float(row_feature['APC']) - float(APC_temp)),
                'field_artillery' : (float(row_feature['field artillery']) - float(field_artillery_temp)),
                'MRL' : (float(row_feature['MRL']) - float(MRL_temp)),
                'drone' : (float(row_feature['drone']) - float(drone_temp)),
            }
            })
            # set the equipment loss values for the current row to be used in the next iteration
            aircraft_temp = row_feature['aircraft']
            helicopter_temp = row_feature['helicopter']
            tank_temp = row_feature['tank']
            APC_temp = row_feature['APC']
            field_artillery_temp = row_feature['field artillery']
            MRL_temp = row_feature['MRL']
            drone_temp = row_feature['drone']
        return result #The result is a dictionary where each key is a date and each value is a dictionary of equipment losses at that date.


def get_feature_personnel():
    result: dict = dict() # create an empty dictionary called result
    with open('../data/russia_losses_personnel.csv', newline='') as csvfile: # open the file russia_losses_personnel.csv
        feature_personnel = csv.DictReader(csvfile) # read the CSV file and create a DictReader object called feature_personnel
        temp = 0
        for row_feature in feature_personnel:  # iterate over each row in the CSV file
            result.update({row_feature['date']: (int(row_feature['personnel'])-int(temp))})  # add a key-value pair to the result dictionary, where the key is the date and the value is the difference between the current number of personnel and the previous number of personnel
            temp = row_feature['personnel'] # update temp to be the current number of personnel
        return result # dictionary containing the difference in personnel for each date.

def get_average_temperature(): #This function is used to calculate the average temperature of four cities - Donetsk, Luhansk, Mariupol, and Melitopol.
    donetsk = get_temperature('../data/donetsk 2022-02-25 to 2023-01-15.csv') # Get the temperature data of Donetsk
    luhansk = get_temperature('../data/luhansk 2022-02-25 to 2023-01-15.csv') # Get the temperature data of Luhansk
    mariupol = get_temperature('../data/mariupol 2022-02-25 to 2023-01-15.csv') # Get the temperature data of Mariupol
    melitopol = get_temperature('../data/melitopol 2022-02-25 to 2023-01-15.csv') # Get the temperature data of Melitopol

    average_temp:dict = dict() # Create an empty dictionary called average_temp to store the average temperature
    size = 4 # Set the size variable to 4 as there are four cities

    #iterate over the temperature data for each city using the 'zip()' function, which allows us to iterate over multiple lists simultaneously.
    for d,l,ma,me in zip(donetsk.items(), luhansk.items(),mariupol.items(),melitopol.items()):
        date = d[0] # extract the date from the temperature data of Donetsk and store it in a variable called 'date'.
        #extract the temperature data of Donetsk, Luhansk, Mariupol, and Melitopol from their respective variables
        #and store them in variables 'd', 'l', 'ma', and 'me' respectively.
        d:dict = d[1]
        l:dict = l[1]
        ma:dict = ma[1]
        me:dict = me[1]
        #calculate the average temperature for the given date by adding the temperatures of all four cities and dividing the sum by the size (which is 4).
        average_temp.update({date: {
            'temp': 1 if round((float(d['temp']) + float(l['temp']) + float(ma['temp']) + float(me['temp'])) / size,
                               2) > 0.5 else 0,
        }}) # If the result of this division is greater than 0.5, the value of 'temp' in the inner dictionary is set to 1. Otherwise, it is set to 0.
    return average_temp # Return the average temperature data (which contains the average temperature for each date.)


def get_temperature(name_file):
    result:dict = dict() # Create an empty dictionary called result
    with open(name_file, newline='') as csvfile: # Open the file specified in the name_file parameter
        city_features = csv.DictReader(csvfile) # Read the CSV file and create a DictReader object called city_features
        size = 4 # Set size to 4
        for row_features in city_features: # Iterate over each row in the CSV file
            # Update the result dictionary
            result.update({row_features['datetime']: {
                'temp':1 if round((float(row_features['temp'])+float(row_features['tempmax'])+float(row_features['tempmin'])+float(row_features['feelslike']))/size,2) > 7 else 0 ,
            }})
        return result # Return the result dictionary

if __name__ == '__main__': #the main function that will be executed when the script is run directly.
    build_data()