import csv
import os
from django.core.management.base import BaseCommand

class Command(BaseCommand):
    help = 'Runs the algorithm and generates CSV files with simulation data'

    def handle(self, *args, **options):
        # Example: Define a list of data dictionaries to output.
        data_sets = [
            {'id': 1, 'value': 10},
            {'id': 2, 'value': 20},
            # … add more or compute dynamically from your algorithm
        ]

        output_dir = os.path.join('simulation_app', 'generated_csv')
        os.makedirs(output_dir, exist_ok=True)

        # Generate multiple CSV files. For example, generating one CSV per dataset.
        for idx, data in enumerate(data_sets):
            file_path = os.path.join(output_dir, f'simulation_data_{idx+1}.csv')
            with open(file_path, mode='w', newline='') as csv_file:
                fieldnames = data.keys()
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
                # In a real algorithm, write rows in a loop:
                writer.writerow(data)
            self.stdout.write(self.style.SUCCESS(f"Generated {file_path}"))
