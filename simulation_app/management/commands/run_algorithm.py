import csv
import os
from django.core.management.base import BaseCommand

class Command(BaseCommand):
    help = 'Runs the algorithm and generates CSV files with simulation data'

    def handle(self, *args, **options):
        # Example: Generate a sample CSV file
        output_dir = os.path.join('simulation_app', 'generated_csv')
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, 'simulation_data.csv')
        
        with open(file_path, mode='w', newline='') as csv_file:
            fieldnames = ['id', 'value']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow({'id': 1, 'value': 100})
        
        self.stdout.write(self.style.SUCCESS(f'Generated {file_path}'))
