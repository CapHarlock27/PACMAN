import datetime
import argparse
from daneel.parameters import Parameters
from daneel.detection import *
from daneel.dream import Dream

def main():
    parser = argparse.ArgumentParser()

    # Primary input argument
    parser.add_argument(
        "-i",
        "--input",
        dest = "input_files",
        type = str,
        nargs = "+",
        required = True,
        help = "Input file(s) to pass. For --transit, provide .yaml parameter files. For --dream, provide path to .pt PyTorch weights.",
    )

    # Modes
    parser.add_argument(
        "-t",
        "--transit",
        dest = "transit",
        required = False,
        help = "Plot the light curve of the selected exoplanet (requires .yaml parameter file(s) )",
        action = "store_true",
    )

    parser.add_argument(
        "--dream",
        dest = "dream",
        required = False,
        help = "'Dream' exoplanetary transit light curve(s) from a trained GAN (requires path to .pt weights)",
        action = "store_true",
    )
    
    parser.add_argument(
        "-d",
        "--detect",
        dest = "detect",
        type = str,                     
        required = False,
        help = "Initialise detection algorithms for Exoplanets (CNN or RF)",
    )

    # parser.add_argument(
    #     "-a",
    #     "--atmosphere",
    #     dest ="complete",
    #     required = False,
    #     help = "Atmospheric Characterisazion from input transmission spectrum",
    #     action = "store_true",
    # )

    # Dedicated Dream arguments
    parser.add_argument(
        "--n_plots",
        dest = "n_plots",
        type = int,
        default = 1,
        help = "Number of samples to generate in Dream mode (default: 1)",
    )

    parser.add_argument(
        "-o",
        "--output",
        dest = "output_file",
        type = str,
        default = "GAN_generated_transits.png",
        help = "Filename to save the generated plot (default: GAN_generated_transits.png)",
    )

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

    elif args.detect is not None:
        print(f"\nUnknown detection method: {args.detect}")
        print("Valid options are: cnn")
    
    elif args.dream:
        weights_path = args.input_files[0]
        n_plots = args.n_plots
        output_file_name = args.output_file
        print(f"Loading weights from: {weights_path}")
        print(f"Generating {n_plots} sample(s)...")
        dreamer = Dream(weights_path, n_plots)
        dreamer.dream(output_file = output_file_name)

    elif args.atmosphere:
        pass
    
    finish = datetime.datetime.now()
    print(f"Daneel finishes at {finish}")

if __name__ == "__main__":
    main()