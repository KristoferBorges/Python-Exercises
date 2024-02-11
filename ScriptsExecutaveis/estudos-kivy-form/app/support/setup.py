import platform
import mysql.connector
from time import sleep
from kivy.clock import Clock
from kivy.core.window import Window

class Setup():
    """
    Classe responsável por configurar o sistema
    """
    def __init__(self):
        if platform.system() == "Windows":
            Window.size = (800, 694)
        else:
            pass

class UserManager:
    """
    Classe responsável por gerenciar o login e logout do usuário
    """
    def __init__(self):
        self.logged_in = False

    def login(self):
        # Lógica de login
        self.logged_in = True

    def logout(self):
        # Lógica de logout
        self.logged_in = False
        

    def is_user_logged_in(self):
        return self.logged_in

class System_Crud:
    """
    Classe responsável por gerenciar o CRUD do sistema
    """
    def __init__(self):
        self.connected = False
        self.conexao = None
        self.conectar_banco()
        self.error = None

    def conectar_banco(self):
        """
        Método responsável por conectar com o banco de dados
        """
        # Conexão com o banco de dados
        try:
            self.conexao = mysql.connector.connect(
                host="localhost",
                user="root",
                password="",
                database="bd_tests",
            )
            self.connected = True
            
        except Exception as erro:
            self.connected = False
            print("Erro ao conectar com o banco de dados: ", erro)

    def createClients(self, RA, nome, semestre, data, comentario=None):
        """
        Método responsável por cadastrar novos clientes
        """
        # Executando teste de inserção
        try:
            # Definição do cursor
            ponteiro = self.conexao.cursor()

            # Query de inserção
            inserir_dados = f"INSERT INTO clientes VALUES ('{RA}', '{nome}', {semestre}, '{data}', '{comentario}')"

            # Executando a query
            ponteiro.execute(inserir_dados)

            # Efetuando o commit
            self.conexao.commit()
            print("Inserção bem-sucedida!")

            return True

        except mysql.connector.errors.IntegrityError as erro:
            self.conexao.rollback()
            self.error = erro
            print(f"Exceção no arquivo Setup: {erro}")
            if "Duplicate entry" in str(erro):
                self.error = "Usuário já cadastrado"
            return False

        except Exception as erro:
            self.conexao.rollback()
            self.error = erro
            print(f"Exceção no arquivo Setup: {erro}")
            return False

        finally:
            print("Fechando conexão com o banco de dados...")
            self.conexao.close()
    
    def createService(self, nome, valor, dependencia):
        """
        Função responsável por cadastrar novos serviços
        """
        # Executando teste de inserção
        try:
            # Definição do cursor
            ponteiro = self.conexao.cursor()

            # Query de inserção
            inserir_dados = f"INSERT INTO servicos VALUES (default, '{nome}', '{valor}', '{dependencia}')"

            # Executando a query
            ponteiro.execute(inserir_dados)

            # Efetuando o commit
            self.conexao.commit()
            print("Inserção bem-sucedida!")

            return True
        
        except mysql.connector.errors.IntegrityError as erro:
            self.conexao.rollback()
            self.error = erro
            print(f"Exceção no arquivo Setup: {erro}")
            if "Duplicate entry" in str(erro):
                self.error = "Serviço já cadastrado"
            return False
        
        except Exception as erro:
            self.conexao.rollback()
            self.error = erro
            print(f"Exceção no arquivo Setup: {erro}")
            return False
        
        finally:
            print("Fechando conexão com o banco de dados...")
            self.conexao.close()

    def registerNewService(self, ra_cliente, id_service, data_registro, data_entrega, valor_cobrado, valor_pendente):
        """
        Método responsável por criar um novo serviço com base nos clientes/serviços já cadastrados
        """
        try:
            init_status = "PENDENTE"

            # Definição do cursor
            ponteiro = self.conexao.cursor()

            # Query de inserção
            inserir_dados = f"INSERT INTO cliente_servico VALUES (default, '{ra_cliente}', '{id_service}', '{data_registro}', '{data_entrega}', '{valor_cobrado}', '{valor_pendente}', '{init_status}')"

            # Executando a query
            ponteiro.execute(inserir_dados)

            # Efetuando o commit
            self.conexao.commit()
            print("Inserção bem-sucedida!")

            return True

        except mysql.connector.errors.IntegrityError as erro:
            self.conexao.rollback()
            self.error = erro
            print(f"Exceção no arquivo Setup: {erro}")
            if "Duplicate entry" in str(erro):
                self.error = "Serviço já cadastrado"
            return False
        
        except Exception as erro:
            self.conexao.rollback()
            self.error = erro
            print(f"Exceção no arquivo Setup: {erro}")
            return False
    
        finally:
            print("Fechando conexão com o banco de dados...")
            self.conexao.close()

    def read_RA(self, ra):
        """
        Método responsável por realizar a busca do RA do cliente
        """
        try:
            self.conectar_banco()
            if self.connected == True:
                ra_cliente = ra
                cursor = self.conexao.cursor()
                cursor.execute(f"SELECT * FROM clientes WHERE ra_cliente = '{ra_cliente}'")
                result = cursor.fetchall()
                if result == []:
                    return False
                else:
                    return result
            else:
                print("Erro ao conectar com o banco de dados!")
        except Exception as erro:
            print(f"Exceção read_RA: {erro}")
        
        finally:
            print("Fechando conexão com o banco de dados...")
            self.conexao.close()

    def read_ID_service(self, id_service):
        """
        Método responsável por realizar a busca do RA do cliente
        """
        try:
            self.conectar_banco()
            if self.connected == True:
                id = id_service
                cursor = self.conexao.cursor()
                cursor.execute(f"SELECT * FROM servicos WHERE id_servico = '{id}'")
                result = cursor.fetchall()
                if result == []:
                    return False
                else:
                    return result
            else:
                print("Erro ao conectar com o banco de dados!")
        except Exception as erro:
            print(f"Exceção read_ID_service: {erro}")
        
        finally:
            print("Fechando conexão com o banco de dados...")
            self.conexao.close()

    def read_ID_orcamento(self, id):
        """
        Método responsável por realizar a busca do ID do orçamento
        """
        try:
            self.conectar_banco()
            if self.connected == True:
                cursor = self.conexao.cursor()
                cursor.execute(f"""SELECT cs.id, c.nome, s.nome, cs.valor_cobrado, cs.data_contratacao, cs.data_entrega, cs.pendencia, cs.situacao FROM clientes c
                                JOIN cliente_servico cs
                                ON c.ra_cliente = cs.id_cliente
                                JOIN servicos s
                                ON cs.tipo_servico = id_servico
                                WHERE cs.id = '{id}'""")
                result = cursor.fetchall()
                if result == []:
                    return False
                else:
                    return result
            
            else:
                print("Erro ao conectar com o banco de dados!")
        
        except Exception as erro:
            print(f"Exceção read_ID_orcamento: {erro}")
        
        finally:
            print("Fechando conexão com o banco de dados...")
            self.conexao.close()

    def read_clients(self, filter=None):
        """
        Método que retorna todos os clientes cadastrados no banco de dados
        """
        try:
            self.conectar_banco()
            if self.connected == True:
                cursor = self.conexao.cursor()
                if filter == None:
                    query = "SELECT * FROM clientes"
                    cursor.execute(query)
                    result = cursor.fetchall()

                elif filter == "RA":
                    query = "SELECT * FROM clientes ORDER BY ra_cliente"
                    cursor.execute(query)
                    result = cursor.fetchall()

                elif filter == "Nome":
                    query = "SELECT * FROM clientes ORDER BY nome"
                    cursor.execute(query)
                    result = cursor.fetchall()

                elif filter == "Semestre":
                    query = "SELECT * FROM clientes ORDER BY semestre"
                    cursor.execute(query)
                    result = cursor.fetchall()

                elif filter == "Registro":
                    query = "SELECT * FROM clientes ORDER BY data_registro"
                    cursor.execute(query)
                    result = cursor.fetchall()

                return result
            else:
                print("Erro ao encontrar clientes!")
                return False
        
        except Exception as erro:
            print(f"Exceção read_clients: {erro}")
        
        finally:
            print("Fechando conexão com o banco de dados...")
            self.conexao.close()

    def read_services(self, filter=None):
        """
        Método que retorna todos os serviços cadastrados no banco de dados
        """
        try:
            self.conectar_banco()
            if self.connected == True:
                cursor = self.conexao.cursor()
                if filter == None:
                    query = "SELECT * FROM servicos"
                    cursor.execute(query)
                    result = cursor.fetchall()

                elif filter == "ID":
                    query = "SELECT * FROM servicos ORDER BY id_servico"
                    cursor.execute(query)
                    result = cursor.fetchall()

                elif filter == "Nome":
                    query = "SELECT * FROM servicos ORDER BY nome"
                    cursor.execute(query)
                    result = cursor.fetchall()

                elif filter == "Valor":
                    query = "SELECT * FROM servicos ORDER BY valor"
                    cursor.execute(query)
                    result = cursor.fetchall()

                elif filter == "Dependencia":
                    query = "SELECT * FROM servicos ORDER BY dependencia"
                    cursor.execute(query)
                    result = cursor.fetchall()

                return result
            else:
                print("Erro ao encontrar serviços!")
                return False
        
        except Exception as erro:
            print(f"Exceção read_services: {erro}")
        
        finally:
            print("Fechando conexão com o banco de dados...")
            self.conexao.close()

    def read_client_service(self, filter=None):
        """
        Método que retorna todos os orçamentos cadastrados no banco de dados
        """
        try:
            self.conectar_banco()
            if self.connected == True:
                cursor = self.conexao.cursor()
                if filter == None:
                    query = """SELECT cs.id, c.nome, s.nome, cs.valor_cobrado, cs.data_contratacao, cs.data_entrega, cs.pendencia, cs.situacao FROM clientes c
                            JOIN cliente_servico cs
                            ON c.ra_cliente = cs.id_cliente
                            JOIN servicos s
                            ON cs.tipo_servico = id_servico"""
                    cursor.execute(query)
                    result = cursor.fetchall()
                
                elif filter == "RA":
                    query = """SELECT cs.id, c.nome, s.nome, cs.valor_cobrado, cs.data_contratacao, cs.data_entrega, cs.pendencia, cs.situacao FROM clientes c
                            JOIN cliente_servico cs
                            ON c.ra_cliente = cs.id_cliente
                            JOIN servicos s
                            ON cs.tipo_servico = id_servico
                            ORDER BY c.ra_cliente"""
                    cursor.execute(query)
                    result = cursor.fetchall()
                
                elif filter == "Nome":
                    query = """SELECT cs.id, c.nome, s.nome, cs.valor_cobrado, cs.data_contratacao, cs.data_entrega, cs.pendencia, cs.situacao FROM clientes c
                            JOIN cliente_servico cs
                            ON c.ra_cliente = cs.id_cliente
                            JOIN servicos s
                            ON cs.tipo_servico = id_servico
                            ORDER BY c.nome"""
                    cursor.execute(query)
                    result = cursor.fetchall()
                
                elif filter == "Serviço":
                    query = """SELECT cs.id, c.nome, s.nome, cs.valor_cobrado, cs.data_contratacao, cs.data_entrega, cs.pendencia, cs.situacao FROM clientes c
                            JOIN cliente_servico cs
                            ON c.ra_cliente = cs.id_cliente
                            JOIN servicos s
                            ON cs.tipo_servico = id_servico
                            ORDER BY s.nome"""
                    cursor.execute(query)
                    result = cursor.fetchall()
                
                elif filter == "ValorCobrado":
                    query = """SELECT cs.id, c.nome, s.nome, cs.valor_cobrado, cs.data_contratacao, cs.data_entrega, cs.pendencia, cs.situacao FROM clientes c
                            JOIN cliente_servico cs
                            ON c.ra_cliente = cs.id_cliente
                            JOIN servicos s
                            ON cs.tipo_servico = id_servico
                            ORDER BY cs.valor_cobrado"""
                    cursor.execute(query)
                    result = cursor.fetchall()
                
                elif filter == "DataContratação":
                    query = """SELECT cs.id, c.nome, s.nome, cs.valor_cobrado, cs.data_contratacao, cs.data_entrega, cs.pendencia, cs.situacao FROM clientes c
                            JOIN cliente_servico cs
                            ON c.ra_cliente = cs.id_cliente
                            JOIN servicos s
                            ON cs.tipo_servico = id_servico
                            ORDER BY cs.data_contratacao"""
                    cursor.execute(query)
                    result = cursor.fetchall()
                
                elif filter == "DataEntrega":
                    query = """SELECT cs.id, c.nome, s.nome, cs.valor_cobrado, cs.data_contratacao, cs.data_entrega, cs.pendencia, cs.situacao FROM clientes c
                            JOIN cliente_servico cs
                            ON c.ra_cliente = cs.id_cliente
                            JOIN servicos s
                            ON cs.tipo_servico = id_servico
                            ORDER BY cs.data_entrega"""
                    cursor.execute(query)
                    result = cursor.fetchall()

                elif filter == "Pendência":
                    query = """SELECT cs.id, c.nome, s.nome, cs.valor_cobrado, cs.data_contratacao, cs.data_entrega, cs.pendencia, cs.situacao FROM clientes c
                            JOIN cliente_servico cs
                            ON c.ra_cliente = cs.id_cliente
                            JOIN servicos s
                            ON cs.tipo_servico = id_servico
                            ORDER BY cs.pendencia"""
                    cursor.execute(query)
                    result = cursor.fetchall()
                
                elif filter == "Status":
                    query = """SELECT cs.id, c.nome, s.nome, cs.valor_cobrado, cs.data_contratacao, cs.data_entrega, cs.pendencia, cs.situacao FROM clientes c
                            JOIN cliente_servico cs
                            ON c.ra_cliente = cs.id_cliente
                            JOIN servicos s
                            ON cs.tipo_servico = id_servico
                            ORDER BY cs.situacao"""
                    cursor.execute(query)
                    result = cursor.fetchall()
                
                return result

            else:
                print("Erro ao encontrar orçamentos!")
                return False
            
        except Exception as erro:
            print(f"Exceção read_client_service: {erro}")
        
        finally:
            print("Fechando conexão com o banco de dados...")
            self.conexao.close()


    def update_client(self, ra, nome, semestre, data, comentario):
        """
        Método responsável por atualizar os dados do cliente
        """
        try:
            self.conectar_banco()
            if self.connected == True:
                cursor = self.conexao.cursor()
                query = f"UPDATE clientes SET nome = '{nome}', semestre = {semestre}, data_registro = '{data}', obs = '{comentario}' WHERE ra_cliente = '{ra}'"
                cursor.execute(query)
                self.conexao.commit()
                print(ra, nome, semestre, data, comentario)
                return True
                    
            else:
                print("Erro ao conectar com o banco de dados!")
                return False
            
        except Exception as erro:
            print(f"Exceção update_client: {erro}")
            self.error = erro
            self.conexao.rollback()
        
        finally:
            print("Fechando conexão com o banco de dados...")
            self.conexao.close()

    def delete(self):
        pass

