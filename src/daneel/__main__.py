import datetime
import argparse
from daneel.parameters import Parameters
from daneel.detection import *


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input",
        dest = "input_files",
        type = str,
        nargs="+",
        required = True,
        help = "Input parameter file(s) to pass",
    )

    parser.add_argument(
        "-t",
        "--transit",
        dest = "transit",
        required = False,        
        help = "Plot the light curve of the selected exoplanet",
        action = "store_true",
    )
    
    parser.add_argument(
        "-d",
        "--detect",
        dest="detect",
        type=str,                     
        required=False,
        help="Initialise detection algorithms for Exoplanets (cnn or rf)",
    )

    # parser.add_argument(
    #     "-a",
    #     "--atmosphere",
    #     dest ="complete",
    #     required = False,
    #     help = "Atmospheric Characterisazion from input transmission spectrum",
    #     action = "store_true",
    # )

    args = parser.parse_args()
    
    """Launch Daneel"""
    start = datetime.datetime.now()
    print(f"Daneel starts at {start}")

    if args.transit:
        
        if len(args.input_files) == 1:
            filename = args.input_files[0]
            input_params = Parameters(filename).params
            transit_section = input_params["transit"]

            model = TransitModel(transit_section)
            model.plot_light_curve()  

        
        else:
            models = []
            for f in args.input_files:
                input_params = Parameters(f).params
                transit_section = input_params["transit"]
                models.append(TransitModel(transit_section))

            TransitModel.plot_multiple_light_curves(models)  
        transit = TransitModel(input_params['transit'])
        transit.plot_light_curve()
    elif args.detect == "cnn":
        from daneel.detection.classifiers import CNNClassifier
        CNN_class = CNNClassifier(input_params)
        CNN_class.run()
    elif args.atmosphere:
        pass
    elif args.detect is not None:
        print(f"\nUnknown detection method: {args.detect}")
        print("Valid options are: cnn")

    finish = datetime.datetime.now()
    print(f"Daneel finishes at {finish}")


if __name__ == "__main__":
    main()
