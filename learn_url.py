import sys


def learn_public_url(port, ssh_username):
    """
    Function to learn a public URL using the provided port and SSH username.
    Parameters:
        port (int): The port number for the URL.
        ssh_username (str): The SSH username used to construct the URL.
    Returns:
        str: The public URL generated based on the input parameters.
    """
    try:
        # Check if SSH username starts with the expected prefix
        if not ssh_username.startswith("s_"):
            raise ValueError("Error: SSH username must start with 's_'.")

        # Extract the username without the prefix
        username = ssh_username.split("_")[1]

        # Check if the port is within the valid range
        if not (1 <= port <= 65535):
            raise ValueError("Error: Port must be an integer between 1 and 65535.")

        # Construct the public URL
        public_url = f"https://{port}-{username}.cloudspaces.litng.ai"
        return public_url

    except IndexError:
        raise ValueError("Error: Missing username after 's_' prefix.")
    except ValueError as e:
        raise ValueError(str(e))


def main():
    # Check if both port and SSH username are provided
    if len(sys.argv) != 3:
        print(f"\nUsage: python {sys.argv[0]} <port> <ssh_username>\n")
        sys.exit("Error: Please provide both port and SSH username.\n")
    port_str, ssh_username = sys.argv[1], sys.argv[2]
    # Attempt to convert port to an integer
    try:
        port_number = int(port_str)
    except ValueError:
        sys.exit("Error: Port must be an integer.")

    try:
        print(learn_public_url(port_number, ssh_username))
    except ValueError as e:
        print(e)
        sys.exit(1)


if __name__ == "__main__":
    main()
