from kivy.uix.popup import Popup
from kivymd.uix.label import MDLabel
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.button import MDFillRoundFlatButton

class FunctionsCase:
    """
    Classe com funções usadas frequentemente no sistema
    """
    def popup_cadastro_sucesso():
        # Pop-up de sucesso
        content = MDBoxLayout(orientation="vertical", padding="10dp")
        label = MDLabel(text="Cadastro realizado com sucesso", halign="center", font_size="15dp", font_name="app/support/fonts/monofonto.otf")
        close_button = MDFillRoundFlatButton(text="Concluir", size_hint=(1, None), font_name="app/support/fonts/monofonto.otf")

        content.add_widget(label)
        content.add_widget(close_button)

        popup = Popup(title="", content=content, size_hint=(0.8, 0.5), auto_dismiss=False)
        close_button.bind(on_release=popup.dismiss)
        popup.open()

    def popup_alteracao_sucesso():
        # Pop-up de sucesso
        content = MDBoxLayout(orientation="vertical", padding="10dp")
        label = MDLabel(text="Alteração realizada com sucesso", halign="center", font_size="15dp", font_name="app/support/fonts/monofonto.otf")
        close_button = MDFillRoundFlatButton(text="Concluir", size_hint=(1, None), font_name="app/support/fonts/monofonto.otf")

        content.add_widget(label)
        content.add_widget(close_button)

        popup = Popup(title="", content=content, size_hint=(0.8, 0.5), auto_dismiss=False)
        close_button.bind(on_release=popup.dismiss)
        popup.open()
    
    def popup_error(error):
        # Pop-up de erro
        content = MDBoxLayout(orientation="vertical", padding="10dp")
        label = MDLabel(text="Erro ao cadastrar cliente", halign="center", font_size="15dp", font_name="app/support/fonts/monofonto.otf")
        close_button = MDFillRoundFlatButton(text="Fechar", size_hint=(1, None), font_name="app/support/fonts/monofonto.otf")

        content.add_widget(label)
        content.add_widget(close_button)

        popup = Popup(title=f"{error}", content=content, size_hint=(0.8, 0.5), auto_dismiss=False)
        close_button.bind(on_release=popup.dismiss)
        popup.open()
    
    def popup_change_error(error):
        # Pop-up de erro
        content = MDBoxLayout(orientation="vertical", padding="10dp")
        label = MDLabel(text="Erro ao realizar alteração", halign="center", font_size="15dp", font_name="app/support/fonts/monofonto.otf")
        close_button = MDFillRoundFlatButton(text="Fechar", size_hint=(1, None), font_name="app/support/fonts/monofonto.otf")

        content.add_widget(label)
        content.add_widget(close_button)

        popup = Popup(title=f"{error}", content=content, size_hint=(0.8, 0.5), auto_dismiss=False)
        close_button.bind(on_release=popup.dismiss)
        popup.open()
    
    def popup_preenchimento():
        # Pop-up de erro de preenchimento
        content = MDBoxLayout(orientation="vertical", padding="10dp")
        label = MDLabel(text="Preencha devidamente todos os campos", halign="center", font_size="15dp", font_name="app/support/fonts/monofonto.otf")
        close_button = MDFillRoundFlatButton(text="Fechar", size_hint=(1, None), font_name="app/support/fonts/monofonto.otf")

        content.add_widget(label)
        content.add_widget(close_button)

        popup = Popup(title="Erro", content=content, size_hint=(0.8, 0.5), auto_dismiss=False)
        close_button.bind(on_release=popup.dismiss)
        popup.open()
    
    def popup_preenchimento_unico():
        # Pop-up de erro de preenchimento
        content = MDBoxLayout(orientation="vertical", padding="10dp")
        label = MDLabel(text="Preencha apenas um dos campos", halign="center", font_size="15dp", font_name="app/support/fonts/monofonto.otf")
        close_button = MDFillRoundFlatButton(text="Fechar", size_hint=(1, None), font_name="app/support/fonts/monofonto.otf")

        content.add_widget(label)
        content.add_widget(close_button)

        popup = Popup(title="Erro", content=content, size_hint=(0.8, 0.5), auto_dismiss=False)
        close_button.bind(on_release=popup.dismiss)
        popup.open()

    def popup_ra_nao_encontrado():
        # Pop-up de RA não encontrado
        content = MDBoxLayout(orientation="vertical", padding="10dp")
        label = MDLabel(text="RA do Cliente não encontrado", halign="center", font_size="15dp", font_name="app/support/fonts/monofonto.otf")
        close_button = MDFillRoundFlatButton(text="Fechar", size_hint=(1, None), font_name="app/support/fonts/monofonto.otf")

        content.add_widget(label)
        content.add_widget(close_button)

        popup = Popup(title="Erro", content=content, size_hint=(0.8, 0.5), auto_dismiss=False)
        close_button.bind(on_release=popup.dismiss)
        popup.open()

    def popup_id_nao_encontrado():
        # Pop-up de ID Service não encontrado
        content = MDBoxLayout(orientation="vertical", padding="10dp")
        label = MDLabel(text="ID Service não encontrado", halign="center", font_size="15dp", font_name="app/support/fonts/monofonto.otf")
        close_button = MDFillRoundFlatButton(text="Fechar", size_hint=(1, None), font_name="app/support/fonts/monofonto.otf")

        content.add_widget(label)
        content.add_widget(close_button)

        popup = Popup(title="Erro", content=content, size_hint=(0.8, 0.5), auto_dismiss=False)
        close_button.bind(on_release=popup.dismiss)
        popup.open()

    def popup_search_error():
        # Pop-up de ID Service não encontrado
        content = MDBoxLayout(orientation="vertical", padding="10dp")
        label = MDLabel(text="Erro ao realizar busca", halign="center", font_size="15dp", font_name="app/support/fonts/monofonto.otf")
        close_button = MDFillRoundFlatButton(text="Fechar", size_hint=(1, None), font_name="app/support/fonts/monofonto.otf")

        content.add_widget(label)
        content.add_widget(close_button)

        popup = Popup(title="Erro", content=content, size_hint=(0.8, 0.5), auto_dismiss=False)
        close_button.bind(on_release=popup.dismiss)
        popup.open()
    
    def popup_preenchimento_invalido():
        # Pop-up de preenchimento inválido
        content = MDBoxLayout(orientation="vertical", padding="10dp")
        label = MDLabel(text="Preencha apenas uma busca por véz", halign="center", font_size="15dp", font_name="app/support/fonts/monofonto.otf")
        close_button = MDFillRoundFlatButton(text="Fechar", size_hint=(1, None), font_name="app/support/fonts/monofonto.otf")

        content.add_widget(label)
        content.add_widget(close_button)

        popup = Popup(title="Erro", content=content, size_hint=(0.8, 0.5), auto_dismiss=False)
        close_button.bind(on_release=popup.dismiss)
        popup.open()

    def filtrandoLetras(texto):
        """
        Método para filtrar apenas letras com ou sem acento de um texto, ignorando números ou caracteres especiais.
        """
        texto_filtrado = ""
        for letra in texto:
            if letra.isalpha():
                texto_filtrado += letra
            elif letra == " ":
                texto_filtrado += letra

        return texto_filtrado.title()

    def formatNome(nome):
        """
        Função que limita o tamanho do nome
        """
        nomeFormatado = ""
        nome = FunctionsCase.filtrandoLetras(nome)
        
        if len(nome) > 15:
            nomeFormatado = nome[:15] + "..."
        else:
            nomeFormatado = nome

        return nomeFormatado