from pyarrow import flight

class DremioBasicServerAuthHandler(flight.ClientAuthHandler):
    """
    ClientAuthHandler for connections to Dremio server endpoint.
    """
    def __init__(self, username, password):
        self.username = username
        self.password = password
        super(flight.ClientAuthHandler, self).__init__()

    def authenticate(self, outgoing, incoming):
        """
        Authenticate with Dremio user credentials.
        """
        basic_auth = flight.BasicAuth(self.username, self.password)
        outgoing.write(basic_auth.serialize())
        self.token = incoming.read()

    def get_token(self):
        """
        Get the token from this AuthHandler.
        """
        return self.token


class BaseDremioService:
    # Método construtor
    def __init__(self, config: dict):
        # Pega as variáveis de conexão ao dremio
        self.hostname = config['DREMIO_HOST']
        self.flightport = config['DREMIO_ARROW_PORT']
        self.username = config['DREMIO_ARROW_USER']
        self.password = config['DREMIO_ARROW_PASS']
        self.tls = False
        self.certs = False

    # Conecta ao dremio e executa a query de extração
    def extract_dremio_dataset(self, sqlquery):
        """
        Connects to Dremio Flight server endpoint with the provided credentials.
        It also runs the query and retrieves the result set.
        """

        try:
            # Default to use an unencrypted TCP connection.
            scheme = "grpc+tcp"
            connection_args = {}

            if self.tls:
                # Connect to the server endpoint with an encrypted TLS connection.
                print('[INFO] Enabling TLS connection')
                scheme = "grpc+tls"
                if self.certs:
                    print('[INFO] Trusted certificates provided')
                    # TLS certificates are provided in a list of connection arguments.
                    with open(self.certs, "rb") as root_certs:
                        connection_args["tls_root_certs"] = root_certs.read()
                else:
                    print('[ERROR] Trusted certificates must be provided to establish a TLS connection')
                    sys.exit()

            client = flight.FlightClient("{}://{}:{}".format(scheme, self.hostname, self.flightport),
                                         **connection_args)

            # Authenticate with the server endpoint.
            client.authenticate(DremioBasicServerAuthHandler(self.username, self.password))
            print('[INFO] Authentication was successful')

            if sqlquery:
                # Construct FlightDescriptor for the query result set.
                flight_desc = flight.FlightDescriptor.for_command(sqlquery)
                print('[INFO] Query: ', sqlquery)

                # Retrieve the schema of the result set.
                schema = client.get_schema(flight_desc)
                print('[INFO] GetSchema was successful')
                print('[INFO] Schema: ', schema)

                # Get the FlightInfo message to retrieve the Ticket corresponding
                # to the query result set.
                flight_info = client.get_flight_info(flight.FlightDescriptor.for_command(sqlquery))
                print('[INFO] GetFlightInfo was successful')
                print('[INFO] Ticket: ', flight_info.endpoints[0].ticket)

                # Retrieve the result set as a stream of Arrow record batches.
                reader = client.do_get(flight_info.endpoints[0].ticket)
                print('[INFO] Reading query results from Dremio')
                df = reader.read_pandas()
                return df

        except Exception as exception:
            print("[ERROR] Exception: {}".format(repr(exception)))
            raise