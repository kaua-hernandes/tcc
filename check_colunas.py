import pandas as pd

arquivo_real = "SPDadosCriminais_2025.xlsx"

print("Verificando se existem outras abas neste arquivo...")
excel_file = pd.ExcelFile(arquivo_real)

print("\nAbas encontradas no arquivo:")
print(excel_file.sheet_names)